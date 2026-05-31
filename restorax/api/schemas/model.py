from typing import Any

from pydantic import BaseModel


class ParamSpecSchema(BaseModel):
    """One tunable restorer parameter, for rendering a typed control in the UI."""
    name: str
    kind: str  # int | float | bool | enum | multiselect
    default: Any
    label: str
    target: str = "extra"  # "param" = top-level RestorerParams field, "extra" = nested in extra
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: list[Any] | None = None
    help: str | None = None


class RestorerInfo(BaseModel):
    name: str
    category: str
    input_color_space: str | None = None
    output_color_space: str | None = None
    requires_temporal: bool | None = None
    min_vram_gb: float | None = None
    scale_factor: int | None = None
    min_ram_gb: float | None = None
    supports_stereo: bool | None = None
    sample_rates: list[int] | None = None
    tags: list[str]
    loaded: bool
    param_schema: list[ParamSpecSchema] = []


class ModelListResponse(BaseModel):
    restorers: list[RestorerInfo]
