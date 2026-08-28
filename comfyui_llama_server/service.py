from __future__ import annotations


class GenerationService:
    def __init__(self, manager, client):
        self.manager = manager
        self.client = client

    def generate(
        self,
        config,
        request,
        stop_after_generate: bool,
        interrupt_check=None,
        on_text=None,
        idle_timeout_seconds: float = 0,
    ):
        self.manager.cancel_idle_stop()
        base_url = self.manager.ensure_started(config, interrupt_check=interrupt_check)
        try:
            return self.client.generate(
                base_url,
                request,
                interrupt_check=interrupt_check,
                on_text=on_text,
            )
        finally:
            if stop_after_generate:
                self.manager.stop()
            else:
                self.manager.schedule_idle_stop(idle_timeout_seconds)
