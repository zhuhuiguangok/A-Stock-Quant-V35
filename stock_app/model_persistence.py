# -*- coding: utf-8 -*-
"""模型持久化管理器 - 优化2（支持AIAlphaEngine多模型保存）"""
import os
import re
import joblib
from pathlib import Path as _Path
import logging
from datetime import datetime
from typing import Dict, Optional, Any, Union

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ModelPersistence:
    """
    模型持久化管理器
    支持：
    - XGBoost模型（joblib）
    - PyTorch模型（state_dict）
    - AIAlphaEngine（自动保存内部所有子模型）
    - 过期检查
    """

    def __init__(self, model_dir: str = 'models', max_age_days: int = 7):
        """
        初始化
        参数:
            model_dir: 模型保存目录
            max_age_days: 模型最大有效天数（超过则视为过期）
        """
        self.model_dir = model_dir
        self.max_age_days = max_age_days
        os.makedirs(model_dir, exist_ok=True)
        logger.info(f"📁 模型持久化目录: {model_dir} (过期天数: {max_age_days})")

    def _get_path(self, name: str, ext: str) -> str:
        """获取完整文件路径（文件名白名单，杜绝路径穿越）"""
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", str(name)):
            raise ValueError(f"illegal model name: {name!r}")
        return os.path.join(self.model_dir, f"{name}.{ext}")

    def _is_fresh(self, filepath: str) -> bool:
        """检查文件是否在有效期内"""
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
            age = (datetime.now() - mtime).days
            return age <= self.max_age_days
        except Exception as e:
            logger.warning(f"检查文件有效期失败: {e}")
            return False

    # ---------- XGBoost ----------
    def save_xgboost(self, model, name: str = 'xgboost'):
        """保存XGBoost模型"""
        try:
            path = self._get_path(name, 'pkl')
            joblib.dump(model, path)
            logger.info(f"💾 XGBoost模型已保存: {path}")
        except Exception as e:
            logger.error(f"❌ 保存XGBoost失败: {e}")

    def load_xgboost(self, name: str = 'xgboost', check_fresh: bool = True):
        """加载XGBoost模型"""
        try:
            path = self._get_path(name, 'pkl')
            if not os.path.exists(path):
                logger.info(f"⏳ XGBoost模型不存在: {path}")
                return None
            if check_fresh and not self._is_fresh(path):
                logger.info(f"⚠️ XGBoost模型已过期（>{self.max_age_days}天），请重新训练")
                return None
            model = joblib.load(path)
            logger.info(f"✅ XGBoost模型加载成功: {path}")
            return model
        except Exception as e:
            logger.error(f"❌ 加载XGBoost失败: {e}")
            return None

    # ---------- PyTorch ----------
    def save_pytorch(self, model, name: str = 'pytorch'):
        """保存PyTorch模型的state_dict"""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch不可用，无法保存模型")
            return
        try:
            path = self._get_path(name, 'pth')
            torch.save(model.state_dict(), path)
            logger.info(f"💾 PyTorch模型已保存: {path}")
        except Exception as e:
            logger.error(f"❌ 保存PyTorch失败: {e}")

    def load_pytorch(self, model, name: str = 'pytorch', check_fresh: bool = True):
        """
        加载PyTorch模型的state_dict
        参数:
            model: 已初始化的模型实例
            name: 模型名称
            check_fresh: 是否检查过期
        返回:
            bool: 是否加载成功
        """
        if not TORCH_AVAILABLE:
            return False
        try:
            path = self._get_path(name, 'pth')
            if not os.path.exists(path):
                logger.info(f"⏳ PyTorch模型不存在: {path}")
                return False
            if check_fresh and not self._is_fresh(path):
                logger.info(f"⚠️ PyTorch模型已过期（>{self.max_age_days}天），请重新训练")
                return False
            model.load_state_dict(torch.load(path))
            logger.info(f"✅ PyTorch模型加载成功: {path}")
            return True
        except Exception as e:
            logger.error(f"❌ 加载PyTorch失败: {e}")
            return False

    # ---------- AIAlphaEngine 多模型支持 ----------
    def save_ai_engine(self, engine, base_name: str = 'ai_engine'):
        """
        保存AIAlphaEngine中的所有模型
        """
        logger.info(f"💾 开始保存AI引擎所有模型，基础名称: {base_name}")

        # 保存普通PyTorch模型
        for model_name, model in engine.models.items():
            full_name = f"{base_name}_{model_name}"
            self.save_pytorch(model, name=full_name)

        # 保存特殊模型（如SmartXGNN）的内部模型
        for spec_name, spec_model in engine.special_models.items():
            if hasattr(spec_model, 'xgb_model') and spec_model.xgb_model is not None:
                self.save_xgboost(spec_model.xgb_model, name=f"{base_name}_{spec_name}_xgb")
            if hasattr(spec_model, 'nn_model') and spec_model.nn_model is not None:
                self.save_pytorch(spec_model.nn_model, name=f"{base_name}_{spec_name}_nn")

        # 持久化模型权重与IC历史：否则重启后回退等权+占位IC，
        # 前端"模型权重分布/IC对比"全部长一个样（2026-09-19 修复）
        try:
            import json as _json
            meta = {
                'saved_at': datetime.now().isoformat(timespec='seconds'),
                'model_weights': {k: float(v) for k, v in getattr(engine, 'model_weights', {}).items()},
                'model_ic_history': {
                    k: [float(x) for x in list(v)[-10:]]
                    for k, v in getattr(engine, 'model_ic_history', {}).items() if v
                },
            }
            _meta_file = _Path(self._get_path(f"{base_name}_meta", 'json'))
            _meta_file.write_text(_json.dumps(meta, ensure_ascii=False, default=str), encoding='utf-8')
            logger.info(f"💾 模型权重/IC元数据已保存: {base_name}_meta.json")
        except Exception as exc:
            logger.warning(f"⚠️ 模型元数据保存失败（不影响模型本体）: {exc}")
        logger.info("✅ AI引擎所有模型保存完成")

    def load_ai_engine(self, engine, base_name: str = 'ai_engine', check_fresh: bool = True):
        """
        加载AIAlphaEngine中的所有模型
        """
        logger.info(f"⏳ 尝试加载AI引擎所有模型，基础名称: {base_name}")
        any_success = False

        for model_name, model in engine.models.items():
            full_name = f"{base_name}_{model_name}"
            success = self.load_pytorch(model, name=full_name, check_fresh=check_fresh)
            if success:
                any_success = True

        for spec_name, spec_model in engine.special_models.items():
            # 【修复】原逻辑 `spec_model.xgb_model is not None` 在冷启动时永远为 False
            # （SmartXGNN.__init__ 把 xgb_model 初始化为 None），导致 XGB 权重永远加载不回来。
            # 正确做法：以持久化文件是否存在为准，存在就加载并注入。
            if hasattr(spec_model, 'xgb_model'):
                xgb_path = self._get_path(f"{base_name}_{spec_name}_xgb", 'pkl')
                if os.path.exists(xgb_path):
                    xgb_model = self.load_xgboost(name=f"{base_name}_{spec_name}_xgb", check_fresh=check_fresh)
                    if xgb_model is not None:
                        spec_model.xgb_model = xgb_model
                        any_success = True
            if hasattr(spec_model, 'nn_model') and spec_model.nn_model is not None:
                success = self.load_pytorch(spec_model.nn_model, name=f"{base_name}_{spec_name}_nn", check_fresh=check_fresh)
                if success:
                    any_success = True

        if any_success:
            logger.info("✅ AI引擎至少有一个模型加载成功")
        else:
            logger.info("AI引擎暂无已训练权重；如需训练，请勾选“训练模型”后开始分析")

        # 恢复权重/IC元数据（与 save_ai_engine 配套；任何模型加载成功都值得恢复）
        try:
            import json as _json
            meta_path = self._get_path(f"{base_name}_meta", 'json')
            if os.path.exists(meta_path):
                with open(meta_path, encoding='utf-8') as fh:
                    meta = _json.load(fh)
                weights = meta.get('model_weights') or {}
                ic_hist = meta.get('model_ic_history') or {}
                if weights:
                    engine.model_weights.update(weights)
                    for name, hist in ic_hist.items():
                        if not engine.model_ic_history.get(name):
                            engine.model_ic_history[name] = list(hist)
                    logger.info(f"♻️ 已恢复模型权重/IC元数据（保存于 {meta.get('saved_at')}）"
                                f"：{ {k: round(v, 3) for k, v in weights.items()} }")
        except Exception as exc:
            logger.warning(f"⚠️ 模型元数据恢复失败（使用默认权重）: {exc}")
        return any_success

    # ---------- InterpretableXGBV18 双模型持久化 ----------
    def save_tree_model(self, model, name: str) -> bool:
        """
        保存InterpretableXGBV18（双模型）
        内部RobustEnsembleModel用pickle序列化
        """
        try:
            import pickle
            path = self._get_path(name, 'pkl')
            if hasattr(model, 'enhanced_model') and model.enhanced_model is not None:
                _Path(path).write_bytes(pickle.dumps(model.enhanced_model, protocol=4))
                logger.info(f"💾 树模型已保存: {path}")
                return True
            else:
                logger.warning(f"⚠️ 树模型 {name} 的enhanced_model为None，跳过保存")
                return False
        except Exception as e:
            logger.error(f"❌ 保存树模型 {name} 失败: {e}")
            return False

    def load_tree_model(self, model, name: str, check_fresh: bool = True) -> bool:
        """
        加载InterpretableXGBV18（双模型）
        成功时将enhanced_model注入到model实例
        """
        try:
            import pickle
            path = self._get_path(name, 'pkl')
            if not os.path.exists(path):
                logger.info(f"⏳ 树模型不存在: {path}")
                return False
            if check_fresh and not self._is_fresh(path):
                logger.info(f"⚠️ 树模型 {name} 已过期（>{self.max_age_days}天），请重新训练")
                return False
            with open(path, 'rb') as f:
                enhanced_model = pickle.load(f)
            model.enhanced_model = enhanced_model
            logger.info(f"✅ 树模型加载成功: {path}")
            return True
        except Exception as e:
            logger.error(f"❌ 加载树模型 {name} 失败: {e}")
            return False

    def save_dual_models(self, trend_model, bottom_model) -> bool:
        """便捷方法：一次保存趋势+抄底双模型"""
        ok1 = self.save_tree_model(trend_model, 'trend_model')
        ok2 = self.save_tree_model(bottom_model, 'bottom_model')
        if ok1 and ok2:
            logger.info("✅ 双模型全部保存成功")
        return ok1 and ok2

    def load_dual_models(self, trend_model, bottom_model, check_fresh: bool = True) -> tuple:
        """便捷方法：一次加载趋势+抄底双模型，返回(trend_ok, bottom_ok)"""
        ok1 = self.load_tree_model(trend_model,  'trend_model',  check_fresh)
        ok2 = self.load_tree_model(bottom_model, 'bottom_model', check_fresh)
        return ok1, ok2


# 全局实例（主流程可导入后直接使用，也可重新配置）
model_persistence = ModelPersistence()
