"""Single-port WebSocket server for SharpA policy deployment."""

from __future__ import annotations

import asyncio
from copy import deepcopy
import logging
import os
from pathlib import Path
import time
from typing import Any, Mapping

from aiohttp import WSMsgType, web
import numpy as np

from dexterity.deploy.template.protocol import (
    ACTION_SCHEMA,
    OBSERVATION_SCHEMA,
    BaseSharpAPolicyAdapter,
    PolicyProtocolError,
    SharpAMetadataFormat,
    build_error_response,
    validate_action,
    validate_metadata_format,
    validate_observation,
)
from dexterity.deploy.template.serialization import packb, unpackb

LOGGER = logging.getLogger("sharpa_policy_deploy_server")
POLICY_HOST = "0.0.0.0"
POLICY_PORT = 5500
POLICY_MAX_MESSAGE_SIZE = 64 * 1024 * 1024


class SharpAPolicyServer:
    """Exchange direct observation/action dicts over one WebSocket connection.

    The robot sends one binary msgpack observation to ``/infer`` and receives
    one binary msgpack action on the same full-duplex connection. Health,
    metadata, and reset operations share the same TCP port.
    """

    def __init__(
        self,
        adapter: BaseSharpAPolicyAdapter,
        *,
        policy_family: str,
        model_name: str | None = None,
        checkpoint_path: str | Path,
        host: str = POLICY_HOST,
        port: int = POLICY_PORT,
        max_message_size: int = POLICY_MAX_MESSAGE_SIZE,
    ) -> None:
        if not policy_family.strip():
            raise ValueError("policy_family must be nonempty")
        resolved_model_name = policy_family if model_name is None else model_name
        configured_model_id = os.environ.get("SHARPA_DEPLOY_MODEL", "").strip()
        if configured_model_id:
            resolved_model_name = configured_model_id
        if not str(resolved_model_name).strip():
            raise ValueError("model_name must be nonempty")
        if not 1 <= int(port) <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if int(max_message_size) < 1:
            raise ValueError("max_message_size must be positive")
        self.adapter = adapter
        self.policy_family = str(policy_family)
        self.model_name = str(resolved_model_name)
        # Public deployment identity. This is exactly the launcher directory
        # name and the task-config run.model value; no alias normalization is
        # allowed at the ROS/server boundary.
        self.model_id = self.model_name
        self.checkpoint_path = str(Path(checkpoint_path).expanduser().absolute())
        self.checkpoint_id = Path(self.checkpoint_path).name
        self.host = str(host)
        self.port = int(port)
        self.max_message_size = int(max_message_size)
        self.task_id = os.environ.get("SHARPA_DEPLOY_TASK", "")
        self.run_id = os.environ.get("SHARPA_DEPLOY_RUN_ID", "")
        self.dataset_path = os.environ.get("SHARPA_DEPLOY_DATASET", "")
        self.default_prompt = os.environ.get("SHARPA_DEPLOY_PROMPT", "")
        self._initial_metadata_format = deepcopy(
            validate_metadata_format(adapter.initial_metadata_format())
        )
        self._active_session_id: str | None = None
        self._active_metadata_format: SharpAMetadataFormat = deepcopy(
            self._initial_metadata_format
        )
        self._client_connected = False
        self._operation_lock: asyncio.Lock | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        metadata = {
            "schema": "sharpa_policy_server.v3",
            "policy_family": self.policy_family,
            "model_id": self.model_id,
            "model_name": self.model_name,
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_path": self.checkpoint_path,
            "task_id": self.task_id,
            "run_id": self.run_id,
            "dataset_path": self.dataset_path,
            "prompt": self.default_prompt,
            "transport": "websocket+binary_msgpack",
            "observation_schema": OBSERVATION_SCHEMA,
            "action_schema": ACTION_SCHEMA,
            "host": self.host,
            "port": self.port,
            "infer_path": "/infer",
            "health_path": "/healthz",
            "metadata_path": "/metadata",
            "reset_path": "/reset",
            "max_message_size": self.max_message_size,
            "image_transport_contract": {
                "encoding": "jpeg",
                "spatial_policy": "preserve_source_resolution",
                "template_resize": False,
                "downstream_spatial_preprocessing_owner": "model",
            },
            "metadata_format": deepcopy(self._initial_metadata_format),
        }
        interface_metadata = dict(self.adapter.interface_metadata())
        if interface_metadata:
            metadata["interface"] = interface_metadata
        return metadata

    def create_app(self) -> web.Application:
        app = web.Application(client_max_size=self.max_message_size)
        app.router.add_get("/healthz", self._health_check)
        app.router.add_get("/metadata", self._metadata_handler)
        app.router.add_post("/reset", self._reset_handler)
        app.router.add_get("/infer", self._infer_handler)
        return app

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        runner = web.AppRunner(self.create_app(), access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        LOGGER.info(
            "serving %s checkpoint=%s on %s:%d",
            self.policy_family,
            self.checkpoint_id,
            self.host,
            self.port,
        )
        try:
            await asyncio.Future()
        finally:
            await runner.cleanup()

    def _lock(self) -> asyncio.Lock:
        if self._operation_lock is None:
            self._operation_lock = asyncio.Lock()
        return self._operation_lock

    async def _health_check(self, request: web.Request) -> web.Response:
        return web.Response(text="OK\n")

    async def _metadata_handler(self, request: web.Request) -> web.Response:
        return web.Response(
            body=packb(self.metadata),
            content_type="application/msgpack",
        )

    async def _reset_adapter(self, session_id: str) -> None:
        async with self._lock():
            await asyncio.to_thread(self.adapter.reset, session_id)
            self._active_session_id = session_id
            self._active_metadata_format = deepcopy(self._initial_metadata_format)

    async def _reset_handler(self, request: web.Request) -> web.Response:
        request_id: int | None = None
        try:
            payload = unpackb(await request.read())
            if not isinstance(payload, Mapping):
                raise PolicyProtocolError("reset request must be a mapping")
            session_id = payload.get("session_id")
            if not isinstance(session_id, str) or not session_id.strip():
                raise PolicyProtocolError("reset.session_id must be a nonempty string")
            raw_request_id = payload.get("request_id")
            if raw_request_id is not None:
                if isinstance(raw_request_id, bool) or not isinstance(
                    raw_request_id, int
                ):
                    raise PolicyProtocolError("reset.request_id must be an integer")
                request_id = raw_request_id
            await self._reset_adapter(session_id)
            response = {
                "schema": "sharpa_policy_reset.v1",
                "session_id": session_id,
                "request_id": request_id,
                "reset": True,
                "metadata_format": deepcopy(self._initial_metadata_format),
            }
            return web.Response(
                body=packb(response), content_type="application/msgpack"
            )
        except PolicyProtocolError as error:
            response = build_error_response(
                code="INVALID_RESET",
                message=str(error),
                request_id=request_id,
                retryable=False,
            )
            return web.Response(
                body=packb(response),
                content_type="application/msgpack",
                status=400,
            )

    async def _infer_handler(self, request: web.Request) -> web.StreamResponse:
        if self._client_connected:
            raise web.HTTPConflict(text="a robot client is already connected\n")
        self._client_connected = True
        self._active_session_id = None
        self._active_metadata_format = deepcopy(self._initial_metadata_format)
        websocket = web.WebSocketResponse(
            autoping=True,
            compress=False,
            max_msg_size=self.max_message_size,
        )
        try:
            await websocket.prepare(request)
            async for message in websocket:
                if message.type == WSMsgType.BINARY:
                    await self._process_observation(websocket, message.data)
                    continue
                if message.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR):
                    break
                error = build_error_response(
                    code="BINARY_MESSAGE_REQUIRED",
                    message="the /infer WebSocket accepts binary msgpack only",
                    request_id=None,
                    retryable=False,
                )
                await websocket.send_bytes(packb(error))
        finally:
            self._client_connected = False
        return websocket

    async def _process_observation(
        self,
        websocket: web.WebSocketResponse,
        message: bytes,
    ) -> None:
        request_id: int | None = None
        try:
            raw_observation = unpackb(message)
            if isinstance(raw_observation, Mapping):
                raw_request_id = raw_observation.get("request_id")
                if isinstance(raw_request_id, (int, np.integer)) and not isinstance(
                    raw_request_id, (bool, np.bool_)
                ):
                    request_id = int(raw_request_id)
            if isinstance(raw_observation, Mapping) and self.default_prompt:
                provided_prompt = raw_observation.get("prompt")
                if provided_prompt not in (None, "", self.default_prompt):
                    raise PolicyProtocolError(
                        "obs.prompt disagrees with the prompt loaded from task dataset"
                    )
                raw_observation = dict(raw_observation)
                raw_observation["prompt"] = self.default_prompt
            raw_session_id = (
                raw_observation.get("session_id")
                if isinstance(raw_observation, Mapping)
                else None
            )
            expected_metadata_format = (
                self._active_metadata_format
                if raw_session_id == self._active_session_id
                else self._initial_metadata_format
            )
            obs = validate_observation(raw_observation, expected_metadata_format)
            session_id = obs["session_id"]
            if session_id != self._active_session_id:
                await self._reset_adapter(session_id)

            started = time.perf_counter()
            async with self._lock():
                raw_action = await asyncio.to_thread(self.adapter.infer, obs)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if not isinstance(raw_action, Mapping):
                raise PolicyProtocolError("adapter.infer must return an action mapping")
            action = dict(raw_action)
            if action.get("session_id") != obs["session_id"]:
                raise PolicyProtocolError(
                    "adapter action session_id does not match the observation"
                )
            if action.get("request_id") != obs["request_id"]:
                raise PolicyProtocolError(
                    "adapter action request_id does not match the observation"
                )
            diagnostics = dict(action.get("diagnostics", {}))
            diagnostics.update(
                {
                    "policy_family": self.policy_family,
                    "checkpoint_id": self.checkpoint_id,
                    "checkpoint_path": self.checkpoint_path,
                    "inference_latency_ms": elapsed_ms,
                }
            )
            action["diagnostics"] = diagnostics
            validate_action(action)
            await websocket.send_bytes(packb(action))
            next_metadata_format = action["next_metadata_format"]
            if next_metadata_format is not None:
                self._active_metadata_format = deepcopy(next_metadata_format)
        except PolicyProtocolError as error:
            response = build_error_response(
                code="INVALID_POLICY_MESSAGE",
                message=str(error),
                request_id=request_id,
                retryable=False,
            )
            await websocket.send_bytes(packb(response))
        except Exception as error:
            LOGGER.exception("policy inference failed request_id=%s", request_id)
            response = build_error_response(
                code="INFERENCE_FAILED",
                message=str(error),
                request_id=request_id,
                retryable=True,
            )
            await websocket.send_bytes(packb(response))
