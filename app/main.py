import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.controller.hello_controller import router as hello_router
from api.controller.surgical_planning_controller import router as surgical_planning_router
from safety.exceptions import GuardrailViolation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

audit_logger = logging.getLogger("safety.audit")

app = FastAPI(title="surgical-planning-ai", version="0.1.0")
app.include_router(hello_router)
app.include_router(surgical_planning_router)


@app.exception_handler(GuardrailViolation)
async def guardrail_violation_handler(
    request: Request, exc: GuardrailViolation
) -> JSONResponse:
    """
    Map any GuardrailViolation raised inside a node or schema validator to
    HTTP 422 with a non-leaky public message and persisting the full
    structured detail to the audit log for compliance review.
    """
    audit_logger.warning(
        "guardrail_violation path=%s method=%s payload=%s",
        request.url.path,
        request.method,
        exc.to_audit_dict(),
    )
    return JSONResponse(
        status_code=422,
        content={
            "error": "guardrail_violation",
            "stage": exc.stage,
            "rule": exc.rule,
            "field": exc.field,
            "message": exc.public_message(),
        },
    )
