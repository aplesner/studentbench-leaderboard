import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from studentbench_leaderboard.llm_client import LLMClient, LLMRequestError


class ClientTests(unittest.TestCase):
    def test_auto_and_explicit_routes(self):
        with patch.dict(os.environ, {}, clear=True):
            client = LLMClient()
            self.assertEqual(client.configuration('gpt-4o')['effective_model'], 'openrouter/openai/gpt-4o')
            self.assertEqual(client.configuration('openai/gpt-5.6-terra')['effective_model'], 'openrouter/openai/gpt-5.6-terra')
            os.environ['OPENAI_API_KEY'] = 'secret'
            self.assertEqual(client.configuration('openai/gpt-5.6-terra')['effective_model'], 'openai/gpt-5.6-terra')
            self.assertEqual(client.configuration('gpt-4o')['provider'], 'openai')
            self.assertEqual(client.configuration('openrouter/openai/gpt-4o')['provider'], 'openrouter')
            os.environ['ANTHROPIC_API_KEY'] = 'other-secret'
            self.assertEqual(client.configuration('claude-sonnet-4')['provider'], 'anthropic')
            self.assertNotIn('secret', json.dumps(client.configuration('gpt-4o')))

    def test_proxy_opt_in_and_redaction(self):
        env = {'USE_LITELLM_PROXY': 'True', 'LITELLM_PROXY_API_BASE': 'https://internal.example/v1', 'LITELLM_PROXY_API_KEY': 'proxy-secret'}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(LLMClient().configuration('gpt-4o')['provider'], 'openrouter')
            config = LLMClient('tailscale').configuration('alias')
            self.assertEqual(config['effective_model'], 'litellm_proxy/alias')
            self.assertNotIn('internal.example', json.dumps(config))
            self.assertNotIn('proxy-secret', json.dumps(config))
            self.assertEqual(LLMClient().configuration('litellm_proxy/alias')['provider'], 'litellm_proxy')

    def test_dispatch_and_no_fallback(self):
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{}'))],
            usage=SimpleNamespace(model_dump=lambda: {'total_tokens': 3}), model='gpt-4o-2024', id='id')
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}, clear=True):
            with patch('litellm.completion', return_value=response) as call:
                raw, meta = LLMClient().complete('gpt-4o', [], {'type': 'object'}, temperature=0)
            self.assertEqual(raw, '{}')
            self.assertEqual(meta['response_model'], 'gpt-4o-2024')
            self.assertEqual(call.call_args.kwargs['model'], 'openai/gpt-4o')
            self.assertEqual(call.call_args.kwargs['api_key'], 'test-key')
            self.assertFalse(call.call_args.kwargs['use_litellm_proxy'])
            self.assertFalse(call.call_args.kwargs['drop_params'])
            self.assertEqual(call.call_args.kwargs['api_base'], 'https://api.openai.com/v1')
            with patch('litellm.completion', side_effect=RuntimeError('failure')) as call:
                with self.assertRaises(RuntimeError):
                    LLMClient().complete('gpt-4o', [], {})
                self.assertEqual(call.call_count, 1)

    def test_proxy_dispatch_and_redacted_error(self):
        env = {'LITELLM_PROXY_API_BASE': 'https://private.example/v1', 'LITELLM_PROXY_API_KEY': 'very-secret'}
        with patch.dict(os.environ, env, clear=True):
            with patch('litellm.completion', side_effect=RuntimeError('very-secret https://private.example/v1')) as call:
                with self.assertRaises(RuntimeError) as caught:
                    LLMClient('tailscale').complete('alias', [], {})
                self.assertEqual(call.call_args.kwargs['model'], 'litellm_proxy/alias')
                self.assertNotIn('very-secret', str(caught.exception))
                self.assertNotIn('private.example', str(caught.exception))

    def test_doctor_auth_is_non_inference(self):
        response = SimpleNamespace(status=200)
        from unittest.mock import MagicMock
        context = MagicMock()
        context.__enter__.return_value = response
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'key'}, clear=True), \
                patch('studentbench_leaderboard.llm_client.urlopen', return_value=context) as call:
            self.assertTrue(LLMClient('openrouter').doctor()['ok'])
        self.assertEqual(call.call_args.args[0].full_url, 'https://openrouter.ai/api/v1/models')
        self.assertEqual(call.call_args.args[0].get_method(), 'GET')

    def test_ambient_litellm_overrides_are_rejected(self):
        import litellm
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'key', 'USE_LITELLM_PROXY': 'true'}, clear=True):
            with self.assertRaisesRegex(ValueError, 'USE_LITELLM_PROXY'):
                LLMClient().complete('gpt-4o', [], {})
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'key'}, clear=True):
            for name, value in [('use_litellm_proxy', True), ('drop_params', True),
                                ('model_fallbacks', ['openrouter/model']),
                                ('model_alias_map', {'openai/gpt-4o': 'other/model'})]:
                with self.subTest(name=name), patch.object(litellm, name, value):
                    with self.assertRaises(ValueError):
                        LLMClient().complete('gpt-4o', [], {})

    def test_installed_sdk_dispatches_anthropic_to_correct_endpoint(self):
        import litellm
        response = litellm.ModelResponse(model='claude-sonnet-4-20250514', choices=[{'message': {'role': 'assistant', 'content': '{}'}}])
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'key'}, clear=True), \
                patch('litellm.main.anthropic_chat_completions.completion', return_value=response) as final:
            raw, meta = LLMClient().complete('anthropic/claude-sonnet-4-20250514', [{'role':'user','content':'hello'}], {'type':'object'})
        self.assertEqual(raw, '{}')
        self.assertEqual(meta['routing']['provider'], 'anthropic')
        self.assertEqual(final.call_args.kwargs['api_base'], 'https://api.anthropic.com/v1/messages')

    def test_installed_sdk_dispatches_openai_and_openrouter(self):
        import litellm
        response = litellm.ModelResponse(model='gpt-4o', choices=[{'message': {'role': 'assistant', 'content': '{}'}}])
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'key', 'OPENROUTER_API_KEY':'key'}, clear=True), \
                patch('litellm.main.openai_chat_completions.completion', return_value=response) as final:
            LLMClient().complete('gpt-4o', [{'role':'user','content':'hello'}], {'type':'object'})
            self.assertEqual(final.call_args.kwargs['api_base'], 'https://api.openai.com/v1')
            with patch('litellm.main.base_llm_http_handler.completion', return_value=response) as router:
                LLMClient().complete('openrouter/openai/gpt-4o', [{'role':'user','content':'hello'}], {'type':'object'})
                self.assertEqual(router.call_args.kwargs['api_base'], 'https://openrouter.ai/api/v1')
                self.assertEqual(router.call_args.kwargs['custom_llm_provider'], 'openrouter')

    def test_installed_sdk_dispatches_explicit_proxy(self):
        import litellm
        response = litellm.ModelResponse(model='alias', choices=[{'message': {'role': 'assistant', 'content': '{}'}}])
        with patch.dict(os.environ, {'LITELLM_PROXY_API_BASE':'https://private.example/v1', 'LITELLM_PROXY_API_KEY':'key'}, clear=True), \
                patch('litellm.main.openai_chat_completions.completion', return_value=response) as final:
            LLMClient('tailscale').complete('alias', [{'role':'user','content':'hello'}], {'type':'object'})
        self.assertEqual(final.call_args.kwargs['api_base'], 'https://private.example/v1')
        self.assertEqual(final.call_args.kwargs['custom_llm_provider'], 'litellm_proxy')
        self.assertEqual(final.call_args.kwargs['model'], 'alias')

    def test_provider_retry_metadata_and_redaction(self):
        class ProviderError(Exception):
            status_code = 429
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'secret'}, clear=True), \
                patch('litellm.completion', side_effect=ProviderError('secret')):
            with self.assertRaises(LLMRequestError) as caught:
                LLMClient().complete('gpt-4o', [], {})
        self.assertEqual(caught.exception.status_code, 429)
        self.assertTrue(caught.exception.transient)
        self.assertEqual(caught.exception.original_type, 'ProviderError')
        self.assertNotIn('secret', str(caught.exception))
        self.assertTrue(caught.exception.__suppress_context__)

    def test_anthropic_doctor_models_endpoint(self):
        from unittest.mock import MagicMock
        context = MagicMock()
        context.__enter__.return_value = SimpleNamespace(status=200)
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'key'}, clear=True), \
                patch('studentbench_leaderboard.llm_client.urlopen', return_value=context) as call:
            self.assertTrue(LLMClient('anthropic').doctor('claude-sonnet-4')['ok'])
        self.assertEqual(call.call_args.args[0].full_url, 'https://api.anthropic.com/v1/models')

    def test_profile_mapping_and_revision_affect_route_identity(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            config = Path(directory) / 'config.toml'
            config.write_text('[profiles.tailscale]\nrevision = "deployment-1"\n'
                              '[profiles.tailscale.models]\n"openai/gpt-5.6-terra" = "judge-alias"\n')
            mapped = LLMClient('tailscale', config).configuration('openai/gpt-5.6-terra')
            self.assertEqual(mapped['requested_model'], 'openai/gpt-5.6-terra')
            self.assertEqual(mapped['mapped_model'], 'judge-alias')
            self.assertEqual(mapped['effective_model'], 'litellm_proxy/judge-alias')
            self.assertEqual(mapped['revision'], 'deployment-1')
            self.assertNotEqual(mapped, LLMClient('tailscale').configuration('openai/gpt-5.6-terra'))
            config.write_text('[profiles.tailscale]\nrevision = "deployment-2"\n'
                              '[profiles.tailscale.models]\n"openai/gpt-5.6-terra" = "judge-alias"\n')
            self.assertNotEqual(mapped, LLMClient('tailscale', config).configuration('openai/gpt-5.6-terra'))
            for invalid in ('revision = 5', 'models = "alias"', '[profiles.tailscale.models]\nfoo = 3'):
                config.write_text('[profiles.tailscale]\n' + invalid if not invalid.startswith('[') else invalid)
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    LLMClient('tailscale', config)

    def test_config_and_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'config.toml'
            config.write_text('default_profile = "openrouter"\n')
            client = LLMClient(config_path=config)
            self.assertEqual(client.configuration('openai/gpt-4o')['provider'], 'openrouter')
            with self.assertRaisesRegex(ValueError, 'Unsupported request settings'):
                client.configuration('gpt-4o', fictional=True)
        with self.assertRaisesRegex(ValueError, 'Unknown connection profile'):
            LLMClient('unknown')

    def test_doctor_missing_key_never_calls_network(self):
        with patch.dict(os.environ, {}, clear=True), patch('studentbench_leaderboard.llm_client.urlopen') as call:
            result = LLMClient('openrouter').doctor()
        self.assertFalse(result['ok'])
        self.assertIn('OPENROUTER_API_KEY', result['error'])
        call.assert_not_called()
