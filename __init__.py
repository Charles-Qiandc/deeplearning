# policy/RDT_repa/__init__.py

# 导入deploy_policy_simple中的函数
try:
    from .deploy_policy_simple import get_model_simple, eval_simple, reset_model, get_model, eval
    print("✅ 成功导入简化版policy函数")
except ImportError as e:
    print(f"⚠️ 导入简化版失败: {e}")
    # 备选：导入标准版本
    try:
        from .deploy_policy import get_model, eval, reset_model
        
        # 创建简化版函数的别名
        get_model_simple = get_model
        eval_simple = eval
        print("✅ 使用标准版policy函数作为备选")
    except ImportError as e2:
        print(f"❌ 无法导入任何policy函数: {e2}")
        raise e2
