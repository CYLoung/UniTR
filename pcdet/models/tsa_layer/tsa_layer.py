import torch.nn as nn
from deformable_attention import DeformableAttention

# class TSALayer(nn.Module):
#     def __init__(self, embed_dim, num_heads, num_levels, num_points):
#         super(TSALayer, self).__init__()
#         self.deformable_attn = DeformableAttention(embed_dim, num_heads, num_levels, num_points)
#         self.norm = nn.LayerNorm(embed_dim)

#     def forward(self, x, prev_x, spatial_shapes, level_start_index, reference_points):
#         # 이전 시퀀스 데이터를 포함하여 deformable attention 적용
#         attn_output = self.deformable_attn(x, prev_x, spatial_shapes, level_start_index, reference_points)
#         x = self.norm(attn_output + x)
#         return x

# class TSALayer(nn.Module):
#     def __init__(self, embed_dim, num_heads, num_levels, num_points):
#         super(TSALayer, self).__init__()
#         self.deformable_attn = DeformableAttention(embed_dim, num_heads, num_levels, num_points)
#         self.norm = nn.LayerNorm(embed_dim)

#     def forward(self, x, prev_x, spatial_shapes, level_start_index, reference_points):
#         # 이전 시퀀스 데이터를 포함하여 deformable attention 적용
#         attn_output = self.deformable_attn(x, prev_x, spatial_shapes, level_start_index, reference_points)
        
#         # 이전 시퀀스의 출력(prev_x)과 현재 시퀀스의 출력(attn_output) 곱셈
#         combined_output = attn_output * prev_x
        
#         # 정규화 및 다음 단계로 전달
#         x = self.norm(combined_output + x)
        
#         return x
class TSALayer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_levels, num_points):
        super(TSALayer, self).__init__()
        print("TEST 1")
        # self.deformable_attn = DeformableAttention(embed_dim, num_heads, num_levels, num_points)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, batch_dict, x, prev_x):
        '''
        batch_dict 는 unitr.py 보면 알 수 있음
        '''
        print("TEST 2")

        # 이전 시퀀스의 출력(prev_x)과 현재 시퀀스의 출력(attn_output) 곱셈
        combined_output = x * prev_x
        
        # 정규화 및 다음 단계로 전달
        x = self.norm(combined_output + x)
        
        return x
