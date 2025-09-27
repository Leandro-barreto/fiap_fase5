#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_monitoring.py
==================
Testes unitários para o módulo monitoring.py
"""
from unittest.mock import MagicMock, patch
import pytest
from fastapi import FastAPI


class TestMonitoringWithPrometheus:
    """Testes para monitoring quando prometheus_fastapi_instrumentator está disponível"""

    def test_setup_monitoring_with_prometheus_available(self):
        """Testa setup_monitoring quando Prometheus está disponível"""
        # Mock do Instrumentator do Prometheus
        mock_instrumentator_instance = MagicMock()
        mock_instrumentator_instance.instrument.return_value = mock_instrumentator_instance
        mock_instrumentator_instance.expose.return_value = mock_instrumentator_instance
        
        mock_instrumentator_class = MagicMock(return_value=mock_instrumentator_instance)
        
        with patch.dict('sys.modules', {'prometheus_fastapi_instrumentator': MagicMock()}):
            with patch('api.monitoring.Instrumentator', mock_instrumentator_class):
                from api.monitoring import setup_monitoring
                
                app = FastAPI()
                setup_monitoring(app)
                
                # Verifica se o Instrumentator foi instanciado
                mock_instrumentator_class.assert_called_once()
                
                # Verifica se instrument foi chamado com a app
                mock_instrumentator_instance.instrument.assert_called_once_with(app)
                
                # Verifica se expose foi chamado com a app
                mock_instrumentator_instance.expose.assert_called_once_with(app)

    def test_instrumentator_chaining(self):
        """Testa se o chaining do Instrumentator funciona corretamente"""
        mock_instrumentator_instance = MagicMock()
        mock_instrumentator_instance.instrument.return_value = mock_instrumentator_instance
        mock_instrumentator_instance.expose.return_value = mock_instrumentator_instance
        
        mock_instrumentator_class = MagicMock(return_value=mock_instrumentator_instance)
        
        with patch.dict('sys.modules', {'prometheus_fastapi_instrumentator': MagicMock()}):
            with patch('api.monitoring.Instrumentator', mock_instrumentator_class):
                from api.monitoring import setup_monitoring
                
                app = FastAPI()
                setup_monitoring(app)
                
                # Verifica a ordem das chamadas (chaining)
                calls = mock_instrumentator_instance.method_calls
                assert len(calls) == 2
                assert calls[0][0] == 'instrument'
                assert calls[1][0] == 'expose'


class TestMonitoringWithoutPrometheus:
    """Testes para monitoring quando prometheus_fastapi_instrumentator não está disponível"""

    def test_setup_monitoring_without_prometheus(self):
        """Testa setup_monitoring quando Prometheus não está disponível (fallback)"""
        # Simula ImportError ao tentar importar prometheus_fastapi_instrumentator
        with patch.dict('sys.modules', {'prometheus_fastapi_instrumentator': None}):
            # Remove o módulo do cache se já foi importado
            import sys
            if 'api.monitoring' in sys.modules:
                del sys.modules['api.monitoring']
            
            # Simula ImportError
            def mock_import(name, *args, **kwargs):
                if name == 'prometheus_fastapi_instrumentator':
                    raise ImportError("No module named 'prometheus_fastapi_instrumentator'")
                return __import__(name, *args, **kwargs)
            
            with patch('builtins.__import__', side_effect=mock_import):
                # Reimporta o módulo para trigger o fallback
                import importlib
                import api.monitoring
                importlib.reload(api.monitoring)
                
                app = FastAPI()
                
                # Deve funcionar sem erro (usando fallback)
                api.monitoring.setup_monitoring(app)
                
                # Verifica se o fallback Instrumentator foi usado
                assert hasattr(api.monitoring, 'Instrumentator')

    def test_fallback_instrumentator_interface(self):
        """Testa se o fallback Instrumentator tem a interface correta"""
        # Force o uso do fallback
        with patch.dict('sys.modules', {'prometheus_fastapi_instrumentator': None}):
            import sys
            if 'api.monitoring' in sys.modules:
                del sys.modules['api.monitoring']
            
            def mock_import(name, *args, **kwargs):
                if name == 'prometheus_fastapi_instrumentator':
                    raise ImportError("No module named 'prometheus_fastapi_instrumentator'")
                return __import__(name, *args, **kwargs)
            
            with patch('builtins.__import__', side_effect=mock_import):
                import importlib
                import api.monitoring
                importlib.reload(api.monitoring)
                
                # Testa a interface do fallback
                instrumentator = api.monitoring.Instrumentator()
                app = FastAPI()
                
                # Deve ter os métodos necessários
                assert hasattr(instrumentator, 'instrument')
                assert hasattr(instrumentator, 'expose')
                
                # Deve retornar self para permitir chaining
                result_instrument = instrumentator.instrument(app)
                assert result_instrument is instrumentator
                
                result_expose = instrumentator.expose(app)
                assert result_expose is instrumentator

    def test_fallback_instrumentator_no_op(self):
        """Testa se o fallback Instrumentator é realmente no-op"""
        with patch.dict('sys.modules', {'prometheus_fastapi_instrumentator': None}):
            import sys
            if 'api.monitoring' in sys.modules:
                del sys.modules['api.monitoring']
            
            def mock_import(name, *args, **kwargs):
                if name == 'prometheus_fastapi_instrumentator':
                    raise ImportError("No module named 'prometheus_fastapi_instrumentator'")
                return __import__(name, *args, **kwargs)
            
            with patch('builtins.__import__', side_effect=mock_import):
                import importlib
                import api.monitoring
                importlib.reload(api.monitoring)
                
                app = FastAPI()
                original_routes = len(app.routes)
                
                # Usar o fallback
                api.monitoring.setup_monitoring(app)
                
                # Como é no-op, não deve adicionar rotas novas
                # (Prometheus normalmente adicionaria /metrics)
                assert len(app.routes) == original_routes


class TestModuleStructure:
    """Testes para estrutura e documentação do módulo"""

    def test_module_docstring_exists(self):
        """Testa se o módulo tem docstring"""
        import api.monitoring
        assert api.monitoring.__doc__ is not None
        assert "Monitoring setup" in api.monitoring.__doc__

    def test_setup_monitoring_function_exists(self):
        """Testa se a função setup_monitoring existe e é chamável"""
        from api.monitoring import setup_monitoring
        assert callable(setup_monitoring)

    def test_setup_monitoring_signature(self):
        """Testa a assinatura da função setup_monitoring"""
        from api.monitoring import setup_monitoring
        import inspect
        
        sig = inspect.signature(setup_monitoring)
        params = list(sig.parameters.keys())
        
        # Deve ter um parâmetro 'app'
        assert 'app' in params
        assert len(params) == 1
        
        # Deve retornar None
        assert sig.return_annotation == None or sig.return_annotation == inspect.Signature.empty


class TestIntegration:
    """Testes de integração do sistema de monitoring"""

    def test_monitoring_integration_with_fastapi(self):
        """Testa integração completa do monitoring com FastAPI"""
        app = FastAPI()
        initial_route_count = len(app.routes)
        
        # Mock do Instrumentator para simular comportamento real
        mock_instrumentator_instance = MagicMock()
        mock_instrumentator_instance.instrument.return_value = mock_instrumentator_instance
        mock_instrumentator_instance.expose.return_value = mock_instrumentator_instance
        
        mock_instrumentator_class = MagicMock(return_value=mock_instrumentator_instance)
        
        with patch.dict('sys.modules', {'prometheus_fastapi_instrumentator': MagicMock()}):
            with patch('api.monitoring.Instrumentator', mock_instrumentator_class):
                from api.monitoring import setup_monitoring
                
                # Configurar monitoring
                setup_monitoring(app)
                
                # Verifica se foi chamado corretamente
                mock_instrumentator_class.assert_called_once()
                mock_instrumentator_instance.instrument.assert_called_once_with(app)
                mock_instrumentator_instance.expose.assert_called_once_with(app)

    def test_multiple_setup_monitoring_calls(self):
        """Testa múltiplas chamadas de setup_monitoring na mesma app"""
        mock_instrumentator_instance = MagicMock()
        mock_instrumentator_instance.instrument.return_value = mock_instrumentator_instance
        mock_instrumentator_instance.expose.return_value = mock_instrumentator_instance
        
        mock_instrumentator_class = MagicMock(return_value=mock_instrumentator_instance)
        
        with patch.dict('sys.modules', {'prometheus_fastapi_instrumentator': MagicMock()}):
            with patch('api.monitoring.Instrumentator', mock_instrumentator_class):
                from api.monitoring import setup_monitoring
                
                app = FastAPI()
                
                # Chamar múltiplas vezes
                setup_monitoring(app)
                setup_monitoring(app)
                
                # Cada chamada deve criar uma nova instância do Instrumentator
                assert mock_instrumentator_class.call_count == 2