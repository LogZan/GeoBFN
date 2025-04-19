
    def forward_update_pos_0(
        self, z, pos, batch, edge_index, edge_length, edge_attr, embed_node=False, **kwargs
    ):

        num_nodes = z.size(0)
        dist, angle, i, j, idx_kj, idx_ji, mask = xyz_to_dat(
            pos,
            edge_index, 
            num_nodes, 
            cutoff=self.cutoff,
            use_torsion=False
        )

        dist_emb, angle_emb = self.emb(dist, angle, idx_kj)
        emb = (dist_emb, angle_emb)
        
        e = self.init_e(z, emb, i, j, edge_attr, embed_node=embed_node)
        v = self.init_v(e, i)
        if torch.isnan(v).any():
            for i in range(50): print("Nan")
            # exit()
        
        for layer_idx, (update_e_layer, update_v_layer) in enumerate(zip(self.update_es, self.update_vs)):
            e = update_e_layer(e, emb, idx_kj, idx_ji, edge_attr)
            v = update_v_layer(e, i)
            e1, e2 = e
            if torch.isnan(v).any():
                for i in range(50): print("Nan")
                # exit()
            
            row, col = edge_index
            edge_vec = pos[row] - pos[col]
            edge_dist = torch.norm(edge_vec, dim=1, keepdim=True) + 1e-8
            edge_dir = edge_vec / edge_dist
            
            # 使用边特征e1和e2来预测位置更新
            # 边特征e1包含了直接相互作用信息，e2包含了更多局部环境信息
            edge_feat_combined = torch.cat([e1, e2], dim=-1)  # [num_edges, 2*hidden_dim]
            
            # 通过MLP预测每条边的更新强度
            update_scale = torch.tanh(self.coord_mlp(edge_feat_combined))  # [num_edges, 1]
            
            # 计算每条边的位置更新向量
            pos_updates = edge_dir * update_scale
            
            # 聚合位置更新: 对每个节点，收集所有相连边的更新
            pos_update = torch.zeros_like(pos)
            
            # 使用scatter_add高效聚合位置更新
            for dim in range(3):  # x, y, z维度
                pos_update[:, dim:dim+1].scatter_add_(0, row.view(-1, 1), -pos_updates[:, dim:dim+1])
                pos_update[:, dim:dim+1].scatter_add_(0, col.view(-1, 1), pos_updates[:, dim:dim+1])
            
            # 通过分组大小归一化更新（类似于消息平均）
            node_degrees = torch.zeros(num_nodes, device=pos.device)
            node_degrees.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
            node_degrees.scatter_add_(0, col, torch.ones_like(col, dtype=torch.float))
            
            node_degrees = torch.clamp(node_degrees, min=1.0).view(-1, 1)  # 避免除零
            pos_update = pos_update / node_degrees
            
            # 应用位置更新
            pos = pos + pos_update
        
        return v, pos

    def forward_update_pos_1(
        self, z, pos, batch, edge_index, edge_length, edge_attr, embed_node=False, **kwargs
    ):
        """
        更新节点特征并基于边特征和角度信息动态更新原子位置
        
        Args:
            z: 节点特征，形状为[num_nodes, hidden_dim]
            pos: 原子坐标，形状为[num_nodes, 3]
            batch: 批处理索引，形状为[num_nodes]
            edge_index: 边索引，形状为[2, num_edges]
            edge_length: 边长度，形状为[num_edges]
            edge_attr: 边特征，形状为[num_edges, edge_dim]
            
        Returns:
            更新后的节点特征和原子坐标
        """
        num_nodes = z.size(0)
        v = None  # 初始化节点特征
        
        # 保存原始坐标用于跟踪更新幅度
        original_pos = pos.clone()
        
        # 多轮消息传递，每轮都重新计算几何特征
        for layer_idx, (update_e_layer, update_v_layer) in enumerate(zip(self.update_es, self.update_vs)):
            # 1. 动态计算最新的几何特征
            dist, angle, i, j, idx_kj, idx_ji, mask = xyz_to_dat(
                pos,  # 使用当前最新位置
                edge_index, 
                num_nodes, 
                cutoff=self.cutoff,
                use_torsion=False
            )
            
            # 2. 根据最新几何特征创建嵌入
            dist_emb, angle_emb = self.emb(dist, angle, idx_kj)
            emb = (dist_emb, angle_emb)
            
            # 3. 如果是第一层，初始化边和节点特征
            if layer_idx == 0:
                e = self.init_e(z, emb, i, j, edge_attr, embed_node=embed_node)
                v = self.init_v(e, i)
            # 否则，更新边和节点特征
            else:
                e = update_e_layer(e, emb, idx_kj, idx_ji, edge_attr)
                v = update_v_layer(e, i)
            
            # 4. 根据最新的特征更新位置
            e1, e2 = e  # e1形状为[num_edges, hidden_dim], e2同样
            
            # 构建边表示，用于预测位置更新
            row, col = edge_index
            edge_vec = pos[row] - pos[col]  # 形状: [num_edges, 3]
            edge_dist = torch.norm(edge_vec, dim=1, keepdim=True) + 1e-8
            edge_dir = edge_vec / edge_dist  # 归一化方向向量
            
            # 使用边特征e1和e2来预测位置更新
            # 边特征e1包含了直接相互作用信息，e2包含了更多局部环境信息
            edge_feat_combined = torch.cat([e1, e2], dim=-1)  # [num_edges, 2*hidden_dim]
            
            # 通过MLP预测每条边的更新强度
            update_scale = torch.tanh(self.coord_mlp(edge_feat_combined))  # [num_edges, 1]
            
            # 计算每条边的位置更新向量
            pos_updates = edge_dir * update_scale
            
            # 聚合位置更新: 对每个节点，收集所有相连边的更新
            pos_update = torch.zeros_like(pos)
            
            # 使用scatter_add高效聚合位置更新
            for dim in range(3):  # x, y, z维度
                pos_update[:, dim:dim+1].scatter_add_(0, row.view(-1, 1), -pos_updates[:, dim:dim+1])
                pos_update[:, dim:dim+1].scatter_add_(0, col.view(-1, 1), pos_updates[:, dim:dim+1])
            
            # 通过分组大小归一化更新（类似于消息平均）
            node_degrees = torch.zeros(num_nodes, device=pos.device)
            node_degrees.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
            node_degrees.scatter_add_(0, col, torch.ones_like(col, dtype=torch.float))
            
            node_degrees = torch.clamp(node_degrees, min=1.0).view(-1, 1)  # 避免除零
            pos_update = pos_update / node_degrees
            
            # 应用位置更新
            pos = pos + pos_update
        
        # 检查位置更新是否超出合理范围
        # pos_shift = torch.norm(pos - original_pos, dim=1).mean()
        # if pos_shift > 1.0:  # 如果平均位移大于1.0埃，可能是更新不稳定
        #     print(f"警告: 位置更新幅度较大 ({pos_shift:.3f})，可能需要调整学习率或更新系数")
        
        return v, pos

    def forward_update_pos_2(
        self, z, pos, batch, edge_index, edge_length, edge_attr, embed_node=False, **kwargs
    ):
        num_nodes = z.size(0)
        pos_updates = torch.zeros_like(pos)

        # 初始化几何特征
        dist, angle, i, j, idx_kj, idx_ji, mask = xyz_to_dat(
            pos, edge_index, num_nodes, 
            cutoff=self.cutoff, use_torsion=False
        )
        
        # 动态更新循环结构
        for layer_idx, (update_e, update_v) in enumerate(zip(self.update_es, self.update_vs)):
            # 使用当前坐标重新计算几何特征
            if layer_idx > 0:
                with torch.no_grad():
                    dist, angle, i, j, idx_kj, idx_ji, mask = xyz_to_dat(
                        pos, edge_index, num_nodes,
                        cutoff=self.cutoff, use_torsion=False
                    )
            
            # 嵌入动态更新的几何特征
            emb = self.emb(dist, angle, idx_kj)
            
            # 初始化/更新边特征
            if layer_idx == 0:
                e = self.init_e(z, emb, i, j, edge_attr, embed_node=embed_node)
            else:
                e = self.update_e(e, emb, idx_kj, idx_ji, edge_attr)  # 需要定义新的update_e层
            
            # 计算坐标更新量
            row, col = edge_index
            edge_vec = pos[row] - pos[col]
            edge_dist = torch.norm(edge_vec, dim=1, keepdim=True) + 1e-8
            edge_dir = edge_vec / edge_dist
            
            # 使用可学习的更新量生成器（新增模块）
            update_magnitude = self.pos_update_net(e)  # [num_edges, 1]
            pos_delta = edge_dir * update_magnitude  # [num_edges, 3]
            
            # 使用散射操作进行高效聚合
            pos_updates = pos_updates.scatter_add(
                0, 
                row.unsqueeze(-1).expand(-1, 3), 
                -pos_delta
            )
            pos_updates = pos_updates.scatter_add(
                0,
                col.unsqueeze(-1).expand(-1, 3),
                pos_delta
            )
            
            # 应用层特定的位置更新
            pos = pos + self.pos_update_coeffs[layer_idx] * pos_updates  # 可学习系数
            
            # 更新节点特征
            v = update_v(e, i)
        
        return v, pos

def forward_update_pos(
        self, z, pos, batch, edge_index, edge_length, edge_attr, embed_node=False, **kwargs
    ):
        num_nodes = z.size(0)
        
        # First compute without updating positions
        dist, angle, i, j, idx_kj, idx_ji, mask = xyz_to_dat(
            pos, edge_index, num_nodes, 
            cutoff=self.cutoff, use_torsion=False
        )
        
        # Initialize with stable embeddings
        dist_emb, angle_emb = self.emb(dist, angle, idx_kj)
        emb = (dist_emb, angle_emb)
        
        # Initialize stable edge and node features
        e = self.init_e(z, emb, i, j, edge_attr, embed_node=embed_node)
        v = self.init_v(e, i)
        
        # Small scaling factor for position updates to prevent large jumps
        pos_scale = 1  # Start with a very small update scale
        
        for layer_idx, (update_e_layer, update_v_layer) in enumerate(zip(self.update_es, self.update_vs)):
            # Safety check on current features
            e1, e2 = e
            if torch.isnan(e1).any() or torch.isnan(e2).any():
                print(f"NaN detected in edge features at layer {layer_idx}, applying fix")
                e1 = torch.nan_to_num(e1, nan=0.0)
                e2 = torch.nan_to_num(e2, nan=0.0)
                e = (e1, e2)
            
            # Recompute geometric features with current positions
            # dist, angle, i, j, idx_kj, idx_ji, mask = xyz_to_dat(
            #     pos, edge_index, num_nodes,
            #     cutoff=self.cutoff, use_torsion=False
            # )
            
            # Create embeddings with safety checks
            # dist_emb, angle_emb = self.emb(dist, angle, idx_kj)
            # if torch.isnan(dist_emb).any():
            #     dist_emb = torch.nan_to_num(dist_emb, nan=0.0)
            # if torch.isnan(angle_emb).any():
            #     angle_emb = torch.nan_to_num(angle_emb, nan=0.0)
            # emb = (dist_emb, angle_emb)
            
            # Update edge and node features with NaN monitoring
            try:
                e = update_e_layer(e, emb, idx_kj, idx_ji, edge_attr)
                v = update_v_layer(e, i)
            except RuntimeError as error:
                print(f"Error in layer {layer_idx}: {error}")
                # Skip position update for this layer
                continue
            
            # Safety check post-update
            if torch.isnan(v).any():
                print(f"NaN detected in node features at layer {layer_idx}")
                # Skip position update for this layer
                continue
            
            # Update positions using edge features
            e1, e2 = e
            row, col = edge_index
            
            # Compute edge vectors with numerical stability
            edge_vec = pos[row] - pos[col]
            edge_dist = torch.norm(edge_vec, dim=1, keepdim=True) + 1e-8
            
            # Safe normalization of edge directions
            edge_dir = edge_vec / edge_dist
            
            # Create combined edge features with clipping to prevent extreme values
            e1_safe = torch.clamp(e1, min=-100, max=100)
            e2_safe = torch.clamp(e2, min=-100, max=100)
            edge_feat_combined = torch.cat([e1_safe, e2_safe], dim=-1)
            
            # Predict position updates with tanh to constrain values
            update_scale = torch.tanh(self.coord_mlp(edge_feat_combined))
            
            # Apply gradually increasing scale factor for position updates
            # This helps stabilize initial updates
            current_pos_scale = pos_scale * (1.0 + layer_idx * 0.5)
            
            # Compute position update vectors
            pos_updates = edge_dir * update_scale * current_pos_scale
            
            # Check for NaN in updates
            if torch.isnan(pos_updates).any():
                print(f"NaN detected in position updates at layer {layer_idx}")
                continue
            
            # Aggregate position updates safely
            pos_update = torch.zeros_like(pos)
            for dim in range(3):
                pos_update[:, dim:dim+1].scatter_add_(0, row.view(-1, 1), -pos_updates[:, dim:dim+1])
                pos_update[:, dim:dim+1].scatter_add_(0, col.view(-1, 1), pos_updates[:, dim:dim+1])
            
            # Normalize updates by node degree
            node_degrees = torch.zeros(num_nodes, device=pos.device)
            node_degrees.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
            node_degrees.scatter_add_(0, col, torch.ones_like(col, dtype=torch.float))
            node_degrees = torch.clamp(node_degrees, min=1.0).view(-1, 1)
            
            # Apply normalized position updates
            pos_update = pos_update / node_degrees
            
            # Clip extreme position updates
            pos_update = torch.clamp(pos_update, min=-0.1, max=0.1)
            
            # Update positions
            pos = pos + pos_update
        
        return v, pos