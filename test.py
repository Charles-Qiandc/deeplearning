#!/usr/bin/env python3
"""
REPA集成验证测试脚本（全局GPU适配版）
测试RDT + REPA对齐损失的完整功能
"""

import sys
import os
import yaml
import torch
import torch.nn.functional as F
import traceback
from pathlib import Path

# ========== 全局 device ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前测试设备: {device}")

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

def test_dinov2_encoder():
    """测试1: DINOv2编码器功能"""
    print("🔍 测试1: DINOv2编码器加载和功能")
    try:
        from models.multimodal_encoder.dinov2_encoder import DinoV2VisionTower, create_dinov2_encoder
        
        dinov2_encoder = create_dinov2_encoder(
            model_size="large", 
            select_feature="patch"
        ).to(device)
        
        dinov2_encoder.print_model_info()
        
        batch_size = 2
        test_images = torch.randn(batch_size, 3, 224, 224, device=device)
        features = dinov2_encoder(test_images)
        print(f"✅ DINOv2前向传播成功")
        print(f"   - 输入形状: {test_images.shape}")
        print(f"   - 输出形状: {features.shape}")
        expected_shape = (batch_size, 256, 1024)
        print(f"   - 期望形状: {expected_shape}")
        assert features.shape == expected_shape, f"形状不匹配: {features.shape} vs {expected_shape}"
        return dinov2_encoder, features.to(device)
    except Exception as e:
        print(f"❌ DINOv2测试失败: {e}")
        traceback.print_exc()
        return None, None

def test_rdt_model():
    """测试2: RDT模型初始化"""
    print("\n🔍 测试2: RDT模型初始化")
    try:
        from models.rdt.model import RDT
        
        rdt_model = RDT(
            output_dim=128,
            horizon=32,
            hidden_size=1152,
            depth=8,
            num_heads=16,
            enable_repa_loss=True,
            repa_activation_layer=4,
            dinov2_feature_dim=1024,
            dtype=torch.bfloat16,
        ).to(device)
        
        print(f"✅ RDT模型初始化成功")
        print(f"   - 总参数量: {sum(p.numel() for p in rdt_model.parameters()):,}")
        print(f"   - Transformer层数: {len(rdt_model.blocks)}")
        print(f"   - REPA启用: {rdt_model.enable_repa_loss}")
        print(f"   - REPA激活层: {rdt_model.repa_activation_layer + 1}")
        print(f"   - 数据类型: {next(rdt_model.parameters()).dtype}")
        
        batch_size = 2
        x = torch.randn(batch_size, 33, 1152, dtype=torch.bfloat16, device=device)
        freq = torch.tensor([25, 30], dtype=torch.long, device=device)
        t = torch.tensor([100, 200], dtype=torch.long, device=device)
        lang_c = torch.randn(batch_size, 512, 1152, dtype=torch.bfloat16, device=device)
        img_c = torch.randn(batch_size, 4096, 1152, dtype=torch.bfloat16, device=device)
        
        pred, activations = rdt_model(x, freq, t, lang_c, img_c)
        print(f"✅ RDT前向传播成功")
        print(f"   - 预测形状: {pred.shape}")
        print(f"   - 中间激活键: {list(activations.keys())}")
        if 'action_tokens_for_repa' in activations:
            action_tokens = activations['action_tokens_for_repa']
            print(f"   - 动作token形状: {action_tokens.shape}")
            assert action_tokens.shape == (batch_size, 32, 1152), f"动作token形状错误: {action_tokens.shape}"
        return rdt_model, activations
    except Exception as e:
        print(f"❌ RDT模型测试失败: {e}")
        traceback.print_exc()
        return None, None

def test_rdt_runner():
    """测试3: RDTRunner完整功能"""
    print("\n🔍 测试3: RDTRunner完整功能")
    try:
        config = {
            'rdt': {
                'hidden_size': 1152,
                'num_heads': 16
            },
            'lang_adaptor': 'linear',
            'img_adaptor': 'linear',
            'state_adaptor': 'linear',
            'noise_scheduler': {
                'num_train_timesteps': 1000,
                'beta_schedule': 'linear',
                'prediction_type': 'epsilon',
                'clip_sample': False,
                'num_inference_timesteps': 50
            }
        }
        from models.rdt_runner import RDTRunner
        rdt_runner = RDTRunner(
            action_dim=128,
            pred_horizon=32,
            config=config,
            lang_token_dim=1024,
            img_token_dim=1024,
            state_token_dim=96,
            max_lang_cond_len=1024,
            img_cond_len=4096,
            enable_repa_loss=True,
            repa_loss_weight=0.2,
            dtype=torch.bfloat16
        ).to(device)
        print(f"✅ RDTRunner初始化成功")
        print(f"   - REPA权重: {rdt_runner.repa_loss_weight}")
        print(f"   - 模型层数: {len(rdt_runner.model.blocks)}")
        if hasattr(rdt_runner.model, 'action_to_vision_projector'):
            print(f"   - 动作投影器已创建")
            test_input = torch.randn(2, 1152, dtype=torch.bfloat16, device=device)
            test_output = rdt_runner.model.action_to_vision_projector(test_input)
            print(f"   - 投影器测试: {test_input.shape} -> {test_output.shape}")
            assert test_output.shape == (2, 1024), f"投影器输出维度错误: {test_output.shape}"
        return rdt_runner
    except Exception as e:
        print(f"❌ RDTRunner测试失败: {e}")
        traceback.print_exc()
        return None

def test_repa_loss_computation(rdt_runner, dinov2_features):
    """测试4: REPA损失计算"""
    print("\n🔍 测试4: REPA损失计算")
    try:
        batch_size = 2
        action_tokens = torch.randn(batch_size, 32, 1152, dtype=torch.bfloat16, device=device)
        vision_features = dinov2_features.to(torch.bfloat16).to(device)
        print(f"   - 动作tokens形状: {action_tokens.shape}")
        print(f"   - 视觉特征形状: {vision_features.shape}")
        repa_loss = rdt_runner.compute_repa_loss(action_tokens, vision_features)
        print(f"✅ REPA损失计算成功: {repa_loss.item():.4f}")
        assert repa_loss.item() <= 0, f"REPA损失应该是负值（负余弦相似度），但得到: {repa_loss.item()}"
        assert repa_loss.item() > -2.0, f"REPA损失过小，可能有问题: {repa_loss.item()}"
        repa_loss_no_vision = rdt_runner.compute_repa_loss(action_tokens, None)
        print(f"✅ 无视觉特征情况: {repa_loss_no_vision.item():.4f}")
        assert repa_loss_no_vision.item() == 0.0, "无视觉特征时应返回0损失"
        rdt_runner.enable_repa_loss = False
        repa_loss_disabled = rdt_runner.compute_repa_loss(action_tokens, vision_features)
        print(f"✅ REPA关闭情况: {repa_loss_disabled.item():.4f}")
        assert repa_loss_disabled.item() == 0.0, "REPA关闭时应返回0损失"
        rdt_runner.enable_repa_loss = True
        return repa_loss.item()
    except Exception as e:
        print(f"❌ REPA损失测试失败: {e}")
        traceback.print_exc()
        return None

def test_full_forward_pass(rdt_runner, dinov2_features):
    """测试5: 完整前向传播和损失计算"""
    print("\n🔍 测试5: 完整前向传播和损失计算")
    try:
        batch_size = 2
        lang_tokens = torch.randn(batch_size, 512, 1024, dtype=torch.bfloat16, device=device)
        lang_attn_mask = torch.ones(batch_size, 512, dtype=torch.bool, device=device)
        img_tokens = torch.randn(batch_size, 4096, 1024, dtype=torch.bfloat16, device=device)
        state_tokens = torch.randn(batch_size, 1, 128, dtype=torch.bfloat16, device=device)
        action_gt = torch.randn(batch_size, 32, 128, dtype=torch.bfloat16, device=device)
        action_mask = torch.ones(batch_size, 1, 64, dtype=torch.bfloat16, device=device)
        ctrl_freqs = torch.tensor([25, 30], dtype=torch.long, device=device)
        vision_features = dinov2_features.to(torch.bfloat16).to(device)
        print(f"   - 视觉特征形状: {vision_features.shape}")
        total_loss, diffusion_loss, repa_loss = rdt_runner.compute_loss(
            lang_tokens=lang_tokens,
            lang_attn_mask=lang_attn_mask,
            img_tokens=img_tokens,
            state_tokens=state_tokens,
            action_gt=action_gt,
            action_mask=action_mask,
            ctrl_freqs=ctrl_freqs,
            vision_features=vision_features
        )
        print(f"✅ 完整前向传播成功")
        print(f"   - 总损失: {total_loss.item():.4f}")
        print(f"   - 扩散损失: {diffusion_loss.item():.4f}")
        print(f"   - REPA损失: {repa_loss.item():.4f}")
        print(f"   - REPA贡献: {(repa_loss.item() * rdt_runner.repa_loss_weight):.4f}")
        print(f"   - 损失比例: {abs(repa_loss.item() * rdt_runner.repa_loss_weight / diffusion_loss.item() * 100):.2f}%")
        assert total_loss.item() > 0, "总损失应为正值"
        assert diffusion_loss.item() > 0, "扩散损失应为正值"
        assert repa_loss.item() <= 0, "REPA损失应为负值或零"
        expected_total = diffusion_loss + rdt_runner.repa_loss_weight * repa_loss
        assert torch.allclose(total_loss, expected_total, atol=1e-5), f"总损失计算错误: {total_loss} vs {expected_total}"
        return True
    except Exception as e:
        print(f"❌ 完整前向传播测试失败: {e}")
        traceback.print_exc()
        return False

def test_gradient_flow(rdt_runner, dinov2_features):
    """测试6: 梯度流验证"""
    print("\n🔍 测试6: 梯度流验证")
    try:
        rdt_runner.train()
        batch_size = 2
        lang_tokens = torch.randn(batch_size, 512, 1024, dtype=torch.bfloat16, device=device, requires_grad=True)
        lang_attn_mask = torch.ones(batch_size, 512, dtype=torch.bool, device=device)
        img_tokens = torch.randn(batch_size, 4096, 1024, dtype=torch.bfloat16, device=device, requires_grad=True)
        state_tokens = torch.randn(batch_size, 1, 128, dtype=torch.bfloat16, device=device, requires_grad=True)
        action_gt = torch.randn(batch_size, 32, 128, dtype=torch.bfloat16, device=device)
        action_mask = torch.ones(batch_size, 1, 64, dtype=torch.bfloat16, device=device)
        ctrl_freqs = torch.tensor([25, 30], dtype=torch.long, device=device)
        vision_features = dinov2_features.to(torch.bfloat16).to(device).detach()
        total_loss, _, _ = rdt_runner.compute_loss(
            lang_tokens=lang_tokens,
            lang_attn_mask=lang_attn_mask,
            img_tokens=img_tokens,
            state_tokens=state_tokens,
            action_gt=action_gt,
            action_mask=action_mask,
            ctrl_freqs=ctrl_freqs,
            vision_features=vision_features
        )
        total_loss.backward()
        components_to_check = [
            ('RDT第4层', rdt_runner.model.blocks[3].norm1.weight),
            ('动作投影器', rdt_runner.model.action_to_vision_projector[0].weight),
            ('最终层', rdt_runner.model.final_layer.ffn_final.fc1.weight),
            ('语言适配器', rdt_runner.lang_adaptor.weight),
        ]
        gradient_ok = True
        for name, param in components_to_check:
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                print(f"✅ {name} 梯度范数: {grad_norm:.6f}")
                if grad_norm < 1e-8:
                    print(f"⚠️  {name} 梯度过小，可能存在梯度消失")
                elif grad_norm > 100:
                    print(f"⚠️  {name} 梯度过大，可能存在梯度爆炸")
            else:
                print(f"❌ {name} 无梯度")
                gradient_ok = False
        if gradient_ok:
            print("✅ 梯度流正常")
        else:
            print("⚠️  部分组件梯度异常")
        return gradient_ok
    except Exception as e:
        print(f"❌ 梯度流测试失败: {e}")
        traceback.print_exc()
        return False

def test_alignment_quality(rdt_runner, dinov2_features):
    """测试7: 对齐质量分析（新增）"""
    print("\n🔍 测试7: 对齐质量分析")
    try:
        batch_size = 2
        horizon = 32
        action_tokens = torch.randn(batch_size, horizon, 1152, dtype=torch.bfloat16, device=device)
        action_tokens_flat = action_tokens.reshape(-1, 1152)
        projected_actions = rdt_runner.model.action_to_vision_projector(action_tokens_flat)
        projected_actions = projected_actions.reshape(batch_size, horizon, 1024)
        projected_actions = F.normalize(projected_actions, dim=-1)
        vision_features_norm = F.normalize(dinov2_features.to(torch.bfloat16).to(device), dim=-1)
        similarities = []
        for b in range(batch_size):
            sim_matrix = torch.mm(projected_actions[b], vision_features_norm[b].t())
            max_sims = sim_matrix.max(dim=1)[0]
            similarities.extend(max_sims.tolist())
        similarities = torch.tensor(similarities, device=device)
        print(f"✅ 相似度统计:")
        print(f"   - 平均值: {similarities.mean():.4f}")
        print(f"   - 标准差: {similarities.std():.4f}")
        print(f"   - 最小值: {similarities.min():.4f}")
        print(f"   - 最大值: {similarities.max():.4f}")
        print(f"   - 中位数: {similarities.median():.4f}")
        if similarities.mean() < -0.5:
            print("⚠️  相似度过低，可能需要调整投影器初始化")
        elif similarities.mean() > 0.5:
            print("⚠️  相似度过高，可能过拟合")
        else:
            print("✅ 相似度分布合理")
        return True
    except Exception as e:
        print(f"❌ 对齐质量测试失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 开始REPA集成测试")
    print("="*50)
    test_results = {}
    dinov2_encoder, dinov2_features = test_dinov2_encoder()
    test_results['dinov2'] = dinov2_encoder is not None
    if not test_results['dinov2']:
        print("❌ DINOv2测试失败，无法继续")
        return
    rdt_model, activations = test_rdt_model()
    test_results['rdt_model'] = rdt_model is not None
    rdt_runner = test_rdt_runner()
    test_results['rdt_runner'] = rdt_runner is not None
    if not test_results['rdt_runner']:
        print("❌ RDTRunner测试失败，无法继续")
        return
    repa_loss = test_repa_loss_computation(rdt_runner, dinov2_features)
    test_results['repa_loss'] = repa_loss is not None
    forward_ok = test_full_forward_pass(rdt_runner, dinov2_features)
    test_results['full_forward'] = forward_ok
    gradient_ok = test_gradient_flow(rdt_runner, dinov2_features)
    test_results['gradient_flow'] = gradient_ok
    alignment_ok = test_alignment_quality(rdt_runner, dinov2_features)
    test_results['alignment_quality'] = alignment_ok
    print("\n" + "="*50)
    print("🏁 测试结果汇总:")
    passed_tests = 0
    total_tests = len(test_results)
    for test_name, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name:20} {status}")
        if passed:
            passed_tests += 1
    print(f"\n📊 总体结果: {passed_tests}/{total_tests} 测试通过")
    if passed_tests == total_tests:
        print("🎉 所有核心测试通过！可以开始集成到训练流程。")
        print("\n📋 下一步操作:")
        print("   1. 修改训练脚本添加DINOv2编码器")
        print("   2. 调整超参数(λ=0.2)")
        print("   3. 运行完整训练测试")
        print("\n⚠️  注意事项:")
        print("   - DINOv2-large 输出 256 个 patches (不是 196)")
        print("   - 投影器维度: 1152 -> 2304 -> 1024")
        print("   - REPA 损失为负值（负余弦相似度）")
    else:
        print("⚠️  部分测试失败，请检查错误信息并修复。")
    return test_results

if __name__ == "__main__":
    main()
