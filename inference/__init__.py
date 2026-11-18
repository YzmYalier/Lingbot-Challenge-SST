"""Online mask client with the submitted judged outline/0.1/2 defaults."""
from .cam_high_masker import CamHighMasker, Sam3StreamClient
from .masked_policy import MaskedPolicyClient, build_policy_client
from submit_paths import config


def create_masker(**overrides):
    q, s = config()["qwen"], config()["inference"]
    settings = dict(qwen_url=f'http://{q["host"]}:{q["port"]}/v1/chat/completions',
                    sam_url=f'http://{s["host"]}:{s["port"]}',
                    outside_scale=0.1, edge_width=2, judge=True)
    settings.update(overrides)
    return CamHighMasker(**settings)


__all__ = ["CamHighMasker", "Sam3StreamClient", "MaskedPolicyClient", "build_policy_client", "create_masker"]
