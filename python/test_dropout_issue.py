"""
测试 dropout 类型错误问题
测试 model_mhc.py 中的 time_matrix dropout 问题
"""

import torch
import torch.nn as nn
import numpy as np


def test_dropout_with_long_tensor():
    """测试 dropout 对 LongTensor 的处理（应该报错）"""
    print("=" * 60)
    print("测试 1: dropout 对 LongTensor（应该报错）")
    print("=" * 60)

    time_matrices = np.array([[0, 1, 2], [3, 4, 5]])
    time_tensor = torch.LongTensor(time_matrices)
    dropout = nn.Dropout(p=0.1)

    print(f"输入类型: {time_tensor.dtype}")

    try:
        result = dropout(time_tensor)
        print("✓ 成功（不应该发生）")
    except RuntimeError as e:
        print(f"✗ 报错: {e}")

    print()


def test_dropout_with_float_tensor():
    """测试 dropout 对 FloatTensor 的处理（应该正常）"""
    print("=" * 60)
    print("测试 2: dropout 对 FloatTensor（应该正常）")
    print("=" * 60)

    time_matrices = np.array([[0, 1, 2], [3, 4, 5]])
    time_tensor = torch.LongTensor(time_matrices).float()
    dropout = nn.Dropout(p=0.1)

    print(f"输入类型: {time_tensor.dtype}")

    try:
        result = dropout(time_tensor)
        print(f"✓ 成功，输出类型: {result.dtype}")
    except RuntimeError as e:
        print(f"✗ 报错: {e}")

    print()


def test_embedding_output_type():
    """测试 Embedding 层的输出类型"""
    print("=" * 60)
    print("测试 3: Embedding 层输出类型")
    print("=" * 60)

    embedding = nn.Embedding(100, 32)

    # 测试 LongTensor 输入
    long_input = torch.LongTensor([[0, 1, 2], [3, 4, 5]])
    long_output = embedding(long_input)

    print(f"输入类型: {long_input.dtype}")
    print(f"输出类型: {long_output.dtype}")

    try:
        dropout = nn.Dropout(p=0.1)
        result = dropout(long_output)
        print(f"✓ dropout 成功，输出类型: {result.dtype}")
    except RuntimeError as e:
        print(f"✗ dropout 报错: {e}")

    print()


def test_clamp_behavior():
    """测试 clamp 后的类型变化"""
    print("=" * 60)
    print("测试 4: clamp 操作后的类型")
    print("=" * 60)

    time_matrices = np.array([[0, 1, 2], [3, 4, 5]])
    time_tensor = torch.LongTensor(time_matrices)

    print(f"clamp 前类型: {time_tensor.dtype}")

    clamped = torch.clamp(time_tensor, min=0, max=10)

    print(f"clamp 后类型: {clamped.dtype}")

    # 测试 clamp 后传入 Embedding
    embedding = nn.Embedding(100, 32)
    embedded = embedding(clamped)

    print(f"Embedding 输出类型: {embedded.dtype}")

    # 测试 Dropout
    dropout = nn.Dropout(p=0.1)
    try:
        result = dropout(embedded)
        print(f"✓ dropout 成功")
    except RuntimeError as e:
        print(f"✗ dropout 报错: {e}")

    print()


def test_original_code_simulation():
    """模拟原始代码（f99e894 之前）"""
    print("=" * 60)
    print("测试 5: 模拟原始代码（无 clamp）")
    print("=" * 60)

    time_matrices = np.array([[0, 1, 2], [3, 4, 5]])

    # 原始代码
    time_tensor = torch.LongTensor(time_matrices)
    embedding = nn.Embedding(100, 32)
    time_matrix = embedding(time_tensor)

    print(f"Embedding 输入类型: {time_tensor.dtype}")
    print(f"Embedding 输出类型: {time_matrix.dtype}")

    dropout = nn.Dropout(p=0.1)
    try:
        result = dropout(time_matrix)
        print(f"✓ dropout 成功，输出类型: {result.dtype}")
    except RuntimeError as e:
        print(f"✗ dropout 报错: {e}")

    print()


def test_clamp_code_simulation():
    """模拟 f99e894 之后的代码（有 clamp 但无 float）"""
    print("=" * 60)
    print("测试 6: 模拟问题代码（有 clamp 但无 .float()）")
    print("=" * 60)

    time_matrices = np.array([[0, 1, 2], [3, 4, 5]])
    time_span = 5

    # 问题代码
    time_tensor = torch.LongTensor(time_matrices)
    time_tensor = torch.clamp(time_tensor, min=0, max=time_span)

    embedding = nn.Embedding(100, 32)
    time_matrix = embedding(time_tensor)

    print(f"clamp 后类型: {time_tensor.dtype}")
    print(f"Embedding 输出类型: {time_matrix.dtype}")

    dropout = nn.Dropout(p=0.1)
    try:
        result = dropout(time_matrix)
        print(f"✓ dropout 成功")
    except RuntimeError as e:
        print(f"✗ dropout 报错: {e}")

    print()


def test_fixed_code_simulation():
    """测试修复后代码（在 Embedding 输出后 .float()）"""
    print("=" * 60)
    print("测试 7: 修复后代码（Embedding 输出后 .float()）")
    print("=" * 60)

    time_matrices = np.array([[0, 1, 2], [3, 4, 5]])
    time_span = 5

    time_tensor = torch.LongTensor(time_matrices)
    time_tensor = torch.clamp(time_tensor, min=0, max=time_span)

    embedding = nn.Embedding(100, 32)
    time_matrix = embedding(time_tensor).float()

    print(f"clamp 后类型（保持 LongTensor）: {time_tensor.dtype}")
    print(f"Embedding 输出后 .float() 类型: {time_matrix.dtype}")

    dropout = nn.Dropout(p=0.1)
    try:
        result = dropout(time_matrix)
        print(f"✓ dropout 成功，输出类型: {result.dtype}")
    except RuntimeError as e:
        print(f"✗ dropout 报错: {e}")

    print()


def test_actual_model_code():
    """测试实际模型代码（模拟 model_mhc.py 的逻辑）"""
    print("=" * 60)
    print("测试 11: 模拟实际模型代码（model_mhc.py）")
    print("=" * 60)

    # 模拟数据
    time_matrices = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    time_span = 10

    # 模拟模型代码
    time_tensor = torch.LongTensor(time_matrices)
    time_tensor = torch.clamp(time_tensor, min=0, max=time_span)

    print(f"clamp 后类型: {time_tensor.dtype}")

    embedding = nn.Embedding(time_span + 1, 32)
    time_matrix_K = embedding(time_tensor)
    time_matrix_V = embedding(time_tensor)

    print(f"Embedding 输出类型: {time_matrix_K.dtype}")

    dropout = nn.Dropout(p=0.1)
    time_matrix_K = dropout(time_matrix_K)
    time_matrix_V = dropout(time_matrix_V)

    print(f"dropout 后类型: {time_matrix_K.dtype}")
    print("✓ 成功")

    print()


def test_alternative_fix():
    """测试替代修复方案（在 Embedding 输出后转 float）"""
    print("=" * 60)
    print("测试 8: 替代修复（Embedding 输出后 .float()）")
    print("=" * 60)

    time_matrices = np.array([[0, 1, 2], [3, 4, 5]])
    time_span = 5

    time_tensor = torch.LongTensor(time_matrices)
    time_tensor = torch.clamp(time_tensor, min=0, max=time_span)

    embedding = nn.Embedding(100, 32)
    time_matrix = embedding(time_tensor).float()

    print(f"Embedding 输出后 .float() 类型: {time_matrix.dtype}")

    dropout = nn.Dropout(p=0.1)
    try:
        result = dropout(time_matrix)
        print(f"✓ dropout 成功，输出类型: {result.dtype}")
    except RuntimeError as e:
        print(f"✗ dropout 报错: {e}")

    print()


def test_training_mode():
    """测试训练模式下 dropout 的行为"""
    print("=" * 60)
    print("测试 9: dropout training vs eval 模式")
    print("=" * 60)

    time_matrices = np.array([[0, 1, 2], [3, 4, 5]])
    time_tensor = torch.LongTensor(time_matrices).float()

    dropout = nn.Dropout(p=0.5)

    print("训练模式 (training=True):")
    dropout.train()
    result_train = dropout(time_tensor)
    print(f"  输出类型: {result_train.dtype}")

    print("评估模式 (training=False):")
    dropout.eval()
    result_eval = dropout(time_tensor)
    print(f"  输出类型: {result_eval.dtype}")

    print()


def test_device_transfer():
    """测试设备传输后的类型"""
    print("=" * 60)
    print("测试 10: 设备传输后的类型")
    print("=" * 60)

    time_matrices = np.array([[0, 1, 2], [3, 4, 5]])
    time_tensor = torch.LongTensor(time_matrices)

    print(f"CPU 上类型: {time_tensor.dtype}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    time_tensor = time_tensor.to(device)

    print(f"GPU 上类型: {time_tensor.dtype}")

    embedding = nn.Embedding(100, 32).to(device)
    time_matrix = embedding(time_tensor)

    print(f"Embedding 输出类型: {time_matrix.dtype}")

    dropout = nn.Dropout(p=0.1).to(device)

    try:
        result = dropout(time_matrix)
        print(f"✓ dropout 成功")
    except RuntimeError as e:
        print(f"✗ dropout 报错: {e}")

    print()


def main():
    print("\n" + "=" * 60)
    print("测试 dropout 类型错误问题")
    print("=" * 60 + "\n")

    tests = [
        ("Dropout 对 LongTensor", test_dropout_with_long_tensor),
        ("Dropout 对 FloatTensor", test_dropout_with_float_tensor),
        ("Embedding 输出类型", test_embedding_output_type),
        ("Clamp 操作后的类型", test_clamp_behavior),
        ("原始代码模拟", test_original_code_simulation),
        ("问题代码模拟", test_clamp_code_simulation),
        ("修复后代码模拟", test_fixed_code_simulation),
        ("替代修复方案", test_alternative_fix),
        ("Training vs Eval 模式", test_training_mode),
        ("设备传输后类型", test_device_transfer),
        ("实际模型代码模拟", test_actual_model_code),
    ]

    for name, test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"✗ 测试 '{name}' 异常: {e}\n")

    print("=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
