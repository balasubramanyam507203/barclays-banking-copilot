from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from fastapi import FastAPI
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
    OTLPMetricExporter,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.instrumentation.fastapi import (
    FastAPIInstrumentor,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)

from app.api.settings import ApiSettings


logger = logging.getLogger(__name__)


@dataclass
class TelemetryRuntime:
    """References to configured telemetry providers."""

    enabled: bool
    tracer_provider: TracerProvider | None = None
    meter_provider: MeterProvider | None = None

    def force_flush(self) -> None:
        """Flushes pending traces and metrics."""

        if self.tracer_provider is not None:
            self.tracer_provider.force_flush()

        if self.meter_provider is not None:
            self.meter_provider.force_flush()


_global_runtime: TelemetryRuntime | None = None


def build_resource(
    settings: ApiSettings,
) -> Resource:
    """Builds OpenTelemetry resource attributes."""

    return Resource.create(
        {
            "service.name": settings.service_name,
            "service.version": settings.version,
            "deployment.environment.name": (
                settings.environment
            ),
        }
    )


def configure_global_providers(
    settings: ApiSettings,
) -> TelemetryRuntime:
    """Configures process-wide trace and metric providers."""

    global _global_runtime

    if _global_runtime is not None:
        return _global_runtime

    resource = build_resource(settings)

    tracer_provider = TracerProvider(
        resource=resource
    )

    metric_readers = []

    if settings.otel_otlp_endpoint:
        os.environ.setdefault(
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            settings.otel_otlp_endpoint,
        )

        tracer_provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter()
            )
        )

        metric_readers.append(
            PeriodicExportingMetricReader(
                OTLPMetricExporter(),
                export_interval_millis=(
                    settings
                    .otel_metric_export_interval_ms
                ),
            )
        )

    if settings.otel_console_exporter:
        tracer_provider.add_span_processor(
            BatchSpanProcessor(
                ConsoleSpanExporter()
            )
        )

        metric_readers.append(
            PeriodicExportingMetricReader(
                ConsoleMetricExporter(),
                export_interval_millis=(
                    settings
                    .otel_metric_export_interval_ms
                ),
            )
        )

    meter_provider = MeterProvider(
        resource=resource,
        metric_readers=metric_readers,
    )

    trace.set_tracer_provider(
        tracer_provider
    )

    metrics.set_meter_provider(
        meter_provider
    )

    _global_runtime = TelemetryRuntime(
        enabled=True,
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
    )

    return _global_runtime


def configure_telemetry(
    application: FastAPI,
    *,
    settings: ApiSettings,
) -> TelemetryRuntime:
    """Configures OpenTelemetry and instruments FastAPI."""

    if not settings.otel_enabled:
        return TelemetryRuntime(
            enabled=False
        )

    runtime = configure_global_providers(
        settings
    )

    excluded_urls = ",".join(
        settings.otel_excluded_urls
    )

    FastAPIInstrumentor.instrument_app(
        application,
        tracer_provider=(
            runtime.tracer_provider
        ),
        meter_provider=(
            runtime.meter_provider
        ),
        excluded_urls=(
            excluded_urls or None
        ),
        exclude_spans=[
            "receive",
            "send",
        ],
    )

    logger.info(
        "OpenTelemetry instrumentation enabled.",
        extra={
            "event": "telemetry_enabled",
        },
    )

    return runtime
