import os
import sys

from django.apps import AppConfig


class StockAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'stock_app'

    def ready(self):
        # 只在实际服务进程中启动自动补数；测试、迁移、静态检查不启动后台线程。
        management_commands_without_scheduler = {
            'test', 'migrate', 'makemigrations', 'collectstatic', 'check', 'shell',
        }
        if any(cmd in sys.argv for cmd in management_commands_without_scheduler):
            return
        if os.environ.get('BULL_BEAR_AUTO_UPDATE', '1').lower() in {'0', 'false', 'no', 'off'}:
            return

        try:
            from .bull_bear_service import start_auto_update_scheduler
            from .tushare_client import get_tushare_token

            interval = int(os.environ.get('BULL_BEAR_AUTO_UPDATE_INTERVAL_SECONDS', '3600'))
            initial_delay = int(os.environ.get('BULL_BEAR_AUTO_UPDATE_INITIAL_DELAY_SECONDS', '60'))
            start_auto_update_scheduler(
                token_provider=get_tushare_token,
                interval_seconds=interval,
                initial_delay_seconds=initial_delay,
            )
        except Exception:
            # 后台调度失败不能影响 Django 正常启动；具体状态会在日志/API里暴露。
            return

        # 模型每周定时重训（默认周六 09:00），失败同样不影响服务启动
        if os.environ.get('MODEL_RETRAIN_AUTO', '1').lower() not in {'0', 'false', 'no', 'off'}:
            try:
                from .model_retrain_service import parse_weekday, start_model_retrain_scheduler

                start_model_retrain_scheduler(
                    weekday=parse_weekday(os.environ.get('MODEL_RETRAIN_WEEKDAY', 'sat')),
                    hour=int(os.environ.get('MODEL_RETRAIN_HOUR', '9')),
                    check_seconds=int(os.environ.get('MODEL_RETRAIN_CHECK_SECONDS', '600')),
                )
            except Exception:
                return

        # AI 研报每日流水线（默认 08:00 一次），失败不影响服务启动
        if os.environ.get('RESEARCH_PIPELINE_AUTO', '1').lower() not in {'0', 'false', 'no', 'off'}:
            try:
                from .research_service import start_pipeline_scheduler

                start_pipeline_scheduler(
                    hour=int(os.environ.get('RESEARCH_PIPELINE_HOUR', '8')),
                    minute=int(os.environ.get('RESEARCH_PIPELINE_MINUTE', '0')),
                    check_seconds=int(os.environ.get('RESEARCH_PIPELINE_CHECK_SECONDS', '600')),
                )
            except Exception:
                return

        # 全球雷达每日更新（默认 08:20，晚于研报流水线以便 AI 晨报引用研报观点）
        if os.environ.get('GLOBAL_RADAR_AUTO', '1').lower() not in {'0', 'false', 'no', 'off'}:
            try:
                from .global_radar_service import start_scheduler

                start_scheduler(
                    hour=int(os.environ.get('GLOBAL_RADAR_HOUR', '8')),
                    minute=int(os.environ.get('GLOBAL_RADAR_MINUTE', '20')),
                    check_seconds=int(os.environ.get('GLOBAL_RADAR_CHECK_SECONDS', '600')),
                )
            except Exception:
                return

        # AI 预测模块每日信号（GBM 主 + Jev/DeepSeek 次要），默认 07:50（收盘后复盘，预测当日已收盘结果）
        if os.environ.get('AI_FORECAST_AUTO', '1').lower() not in {'0', 'false', 'no', 'off'}:
            try:
                from .forecast_service import start_forecast_scheduler

                start_forecast_scheduler(
                    hour=int(os.environ.get('AI_FORECAST_HOUR', '7')),
                    minute=int(os.environ.get('AI_FORECAST_MINUTE', '50')),
                    check_seconds=int(os.environ.get('AI_FORECAST_CHECK_SECONDS', '600')),
                )
            except Exception:
                return

        # 板块预测每日 Top10（GBM 主 + Jev 语义化次要），默认 08:00（收盘后）
        if os.environ.get('SECTOR_FORECAST_AUTO', '1').lower() not in {'0', 'false', 'no', 'off'}:
            try:
                from .sector_forecast_service import start_sector_forecast_scheduler

                start_sector_forecast_scheduler(
                    hour=int(os.environ.get('SECTOR_FORECAST_HOUR', '8')),
                    minute=int(os.environ.get('SECTOR_FORECAST_MINUTE', '0')),
                    check_seconds=int(os.environ.get('SECTOR_FORECAST_CHECK_SECONDS', '600')),
                )
            except Exception:
                return
