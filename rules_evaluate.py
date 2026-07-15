import json
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import re

class GeologicalRelationProcessor:
   
    def __init__(self):
        self.geological_knowledge = {
            'stratigraphic_sequence': {
                '太古代': 0, '震旦系': 1, '震旦纪': 1, '寒武纪': 2, '奥陶纪': 3,
                '石炭': 4, '二迭纪': 5, '三叠纪': 6, '侏罗纪': 7, '白垩纪': 8,
                '古生代': 2.5, '中生代': 7.5, '新生代': 9, '第四纪': 10,
                '下古生代': 2, '上古生代': 4
            }
        }
        

        self.valid_combinations = {
            'IN': [('TOP', 'GS'), ('TOP', 'TOP'), ('GS', 'GS'), ('STR', 'STR'), 
                   ('STR', 'GS'), ('GS', 'STR'), ('ROC', 'GS'), ('GS', 'ROC'),
                   ('TOP', 'STR'), ('STR', 'TOP')],
            'PO': [('GS', 'GS'), ('GS', 'TOP'), ('TOP', 'GS')],
            'N': [('GS', 'STR'), ('GS', 'TOP'), ('TOP', 'TOP'), ('STR', 'STR'), 
                  ('GS', 'GS'), ('TOP', 'STR'), ('STR', 'TOP')],
            'S': [('GS', 'STR'), ('GS', 'TOP'), ('TOP', 'TOP'), ('TOP', 'GS'),('STR', 'STR'),
                  ('GS', 'GS'), ('TOP', 'STR'), ('STR', 'TOP')],
            'E': [('GS', 'TOP'), ('TOP', 'TOP'), ('GS', 'GS'), ('TOP', 'GS')],
            'W': [('GS', 'TOP'), ('TOP', 'TOP'), ('GS', 'GS'), ('TOP', 'GS')],
            'WN': [('GS', 'TOP'), ('TOP', 'TOP'), ('GS', 'GS'), ('TOP', 'GS')],
            'WS': [('GS', 'TOP'), ('TOP', 'TOP'), ('GS', 'GS'), ('TOP', 'GS')],
            'EN': [('GS', 'TOP'), ('TOP', 'TOP'), ('GS', 'GS'), ('TOP', 'GS')],
            'ES': [('GS', 'TOP'), ('TOP', 'TOP'), ('GS', 'GS'), ('TOP', 'GS')],
            'C': [('GS', 'STR'), ('GS', 'TOP'), ('STR', 'STR'), ('TOP', 'TOP')],
            'A': [('STR', 'STR'), ('GS', 'STR'), ('GS', 'GS')],
            'BW': [('STR', 'STR'), ('GS', 'STR'), ('GS', 'GS')]
        }
        self.geological_anomaly_keywords = ['倒转', '逆冲', '推覆', '侵入']
    
    def load_data(self, file_path: str) -> List[Dict]:
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
        return data
    
    def load_text_data(self, file_path: str) -> List[Dict]:
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())

                    if 'text' in item:
                        data.append(item)
        return data
    
    def diagnose_and_correct(self, spo_list: List[Dict], text: str = None) -> Tuple[List[Dict], List[Dict]]:
        corrected_spo = spo_list.copy()
        issues = []
        corrected_spo, dup_issues = self._remove_duplicates(corrected_spo)
        issues.extend(dup_issues)
        corrected_spo, type_issues = self._fix_type_compatibility(corrected_spo)
        issues.extend(type_issues)
        corrected_spo, direction_issues = self._fix_direction_conflicts(corrected_spo)
        issues.extend(direction_issues)
        corrected_spo, strat_issues = self._fix_stratigraphic_logic(corrected_spo, text)
        issues.extend(strat_issues)
        if text:
            corrected_spo, text_issues = self._check_entities_in_text(corrected_spo, text)
            issues.extend(text_issues)
        
        return corrected_spo, issues
    
    def _check_entities_in_text(self, spo_list: List[Dict], text: str) -> Tuple[List[Dict], List[Dict]]:
        if not text or not isinstance(text, str):
            return spo_list, []
        
        corrected_spo = []
        issues = []
        
        for spo in spo_list:
            subject_value = spo['subject']['value']
            object_value = spo['object']['value']
            

            new_subject_value, sub_match_status, sub_matched_str = self._verify_and_correct_entity(subject_value, text)

            new_object_value, obj_match_status, obj_matched_str = self._verify_and_correct_entity(object_value, text)
            
            is_valid = True
            missing_entities = []
            corrections = []

            if sub_match_status == 'failed':
                is_valid = False
                missing_entities.append(f"主体'{subject_value}'")
            elif sub_match_status == 'corrected':
                corrections.append(f"主体'{subject_value}'修正为'{new_subject_value}'")
                spo['subject']['value'] = new_subject_value

            if obj_match_status == 'failed':
                is_valid = False
                missing_entities.append(f"客体'{object_value}'")
            elif obj_match_status == 'corrected':
                corrections.append(f"客体'{object_value}'修正为'{new_object_value}'")
                spo['object']['value'] = new_object_value
            
            if not is_valid:
                issues.append({
                    'type': '实体幻觉(无法溯源)',
                    'relation': spo,
                    'missing_entities': missing_entities,
                    'action': '已删除该关系'
                })
            else:
                corrected_spo.append(spo)
                if corrections:
                    issues.append({
                        'type': '实体幻觉(模糊匹配修正)',
                        'original_relation': spo, 
                        'corrections': corrections,
                        'action': '已修正实体名称'
                    })
        
        return corrected_spo, issues

    def _verify_and_correct_entity(self, entity: str, text: str) -> Tuple[str, str, str]:

        if not entity or not text:
            return entity, 'failed', ""

        # 第一级：严格子串匹配
        if entity in text:
            return entity, 'matched', entity
        
        # 第二级：基于编辑距离的模糊匹配
        best_match_str, max_similarity = self._fuzzy_match_entity(entity, text)
        
        if max_similarity > 0.8:
            return best_match_str, 'corrected', best_match_str
        
        return entity, 'failed', ""

    def _fuzzy_match_entity(self, entity: str, text: str) -> Tuple[str, float]:

        if not entity or not text:
            return "", 0.0
        
        ent_len = len(entity)
        if ent_len == 0:
            return "", 0.0
            
        min_win_len = max(1, int(ent_len * 0.8))
        max_win_len = int(ent_len * 1.2) + 1
        
        best_sim = 0.0
        best_substring = ""
        

        for win_len in range(min_win_len, min(max_win_len, len(text) + 1)):

            for i in range(len(text) - win_len + 1):
                substring = text[i:i+win_len]
                sim = self._calculate_similarity(entity, substring)
                if sim > best_sim:
                    best_sim = sim
                    best_substring = substring
                    
        return best_substring, best_sim

    def _calculate_similarity(self, str1: str, str2: str) -> float:

        if not str1 and not str2:
            return 1.0
        if not str1 or not str2:
            return 0.0
            
        dist = self._edit_distance(str1, str2)
        max_len = max(len(str1), len(str2))
        
        if max_len == 0:
            return 1.0
            
        return 1.0 - (dist / max_len)

    def _edit_distance(self, str1: str, str2: str) -> int:

        m, n = len(str1), len(str2)
        # 创建DP表
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
            
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i-1] == str2[j-1]:
                    cost = 0
                else:
                    cost = 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,      
                    dp[i][j-1] + 1,      
                    dp[i-1][j-1] + cost  
                )
                
        return dp[m][n]

    def _is_entity_in_text(self, entity: str, text: str) -> bool:
        _, status, _ = self._verify_and_correct_entity(entity, text)
        return status != 'failed'
    
    def _clean_text(self, text: str) -> str:

        if not text:
            return ""
        

        text = re.sub(r'\s+', '', text)  #
        text = re.sub(r'[，。；：、！？"\'《》【】()（）]', '', text)  
        return text
    
    def _extract_stratigraphic_core(self, entity: str) -> str:

        if not entity:
            return ""
        

        separators = ['-', '—', '~', '～', '、', '和', '与', '及']
        

        for sep in separators:
            if sep in entity:
                parts = entity.split(sep)
                if parts[0].strip():
                    return parts[0].strip()
        
        return entity.strip()
    
    def _remove_duplicates(self, spo_list: List[Dict]) -> Tuple[List[Dict], List[Dict]]:

        seen = set()
        unique_spo = []
        issues = []
        
        for spo in spo_list:

            key = (
                spo['predicate'],
                spo['subject']['value'],
                spo['subject']['type'],
                spo['object']['value'],
                spo['object']['type']
            )
            
            if key not in seen:
                seen.add(key)
                unique_spo.append(spo)
            else:
                issues.append({
                    'type': '重复关系',
                    'relation': f"{spo['subject']['value']} --{spo['predicate']}--> {spo['object']['value']}",
                    'action': '已删除重复项'
                })
        
        return unique_spo, issues
    
    def _fix_type_compatibility(self, spo_list: List[Dict]) -> Tuple[List[Dict], List[Dict]]:

        corrected_spo = []
        issues = []
        
        for spo in spo_list:
            pred = spo['predicate']
            subj_type = spo['subject']['type']
            obj_type = spo['object']['type']
            
            if pred in self.valid_combinations:
                valid = False
                for valid_combo in self.valid_combinations[pred]:
                    if subj_type == valid_combo[0] and obj_type == valid_combo[1]:
                        valid = True
                        break
                
                if not valid:
                    corrected = self._suggest_type_correction(spo)
                    if corrected:
                        corrected_spo.append(corrected)
                        issues.append({
                            'type': '实体类型与关系不兼容',
                            'original': spo,
                            'corrected': corrected,
                            'reason': f'关系{pred}不支持{subj_type}->{obj_type}的组合'
                        })
                    else:
                        corrected_spo.append(spo)
                        issues.append({
                            'type': '实体类型与关系不兼容（无法修正）',
                            'relation': spo,
                            'reason': f'关系{pred}不支持{subj_type}->{obj_type}的组合'
                        })
                else:
                    corrected_spo.append(spo)
            else:
                corrected_spo.append(spo)
        
        return corrected_spo, issues
    
    def _suggest_type_correction(self, spo: Dict) -> Dict:
        pred = spo['predicate']
        subj_type = spo['subject']['type']
        obj_type = spo['object']['type']
        

        corrections = {

            ('IN', 'STR', 'GS'): ('GS', 'STR'),  
            ('IN', 'ROC', 'GS'): ('GS', 'ROC'),  
            ('PO', 'TOP', 'GS'): ('GS', 'GS'),   
            ('N', 'STR', 'GS'): ('GS', 'STR'),  
            ('S', 'STR', 'GS'): ('GS', 'STR'),  

        ('IN', 'TOP', 'ROC'): ('ROC', 'TOP'),  
        ('IN', 'GS', 'TOP'): ('TOP', 'GS'),   
        ('IN', 'TOP', 'STR'): ('STR', 'TOP'),  
        ('IN', 'ROC', 'STR'): ('STR', 'ROC'),  
        ('IN', 'STR', 'ROC'): ('ROC', 'STR'),  
        ('IN', 'TOP', 'GS'): ('GS', 'TOP'),    
        

        ('PO', 'STR', 'ROC'): ('ROC', 'STR'), 
        ('PO', 'STR', 'GS'): ('GS', 'STR'), 
        ('PO', 'ROC', 'TOP'): ('TOP', 'ROC'), 
        ('PO', 'GS', 'ROC'): ('ROC', 'GS'), 
        ('PO', 'GS', 'STR'): ('STR', 'GS'), 
        ('PO', 'TOP', 'ROC'): ('ROC', 'TOP'),  
        

        ('N', 'TOP', 'ROC'): ('ROC', 'TOP'),  
        ('N', 'TOP', 'STR'): ('STR', 'TOP'), 
        ('S', 'TOP', 'ROC'): ('ROC', 'TOP'),  
        ('S', 'TOP', 'STR'): ('STR', 'TOP'),  
        ('E', 'STR', 'ROC'): ('ROC', 'STR'),  
        ('W', 'STR', 'ROC'): ('ROC', 'STR'), 
        ('EN', 'TOP', 'GS'): ('GS', 'TOP'),  
        ('WS', 'GS', 'TOP'): ('TOP', 'GS'),   
        

        ('C', 'TOP', 'ROC'): ('ROC', 'TOP'),  
        ('C', 'TOP', 'STR'): ('STR', 'TOP'),   
        ('C', 'TOP', 'GS'): ('GS', 'TOP'),     
        ('C', 'ROC', 'GS'): ('GS', 'ROC'),     
        ('C', 'STR', 'GS'): ('GS', 'STR'),    
        ('C', 'ROC', 'STR'): ('STR', 'ROC'),   
        

        ('A', 'TOP', 'STR'): ('STR', 'TOP'),   
        ('A', 'TOP', 'ROC'): ('ROC', 'TOP'),  
        ('A', 'GS', 'TOP'): ('TOP', 'GS'),    
        ('A', 'ROC', 'TOP'): ('TOP', 'ROC'), 
        ('A', 'STR', 'TOP'): ('TOP', 'STR'),   
        

        ('BW', 'TOP', 'ROC'): ('ROC', 'TOP'),  
        ('BW', 'TOP', 'STR'): ('STR', 'TOP'), 
        ('BW', 'TOP', 'GS'): ('GS', 'TOP'),    
        ('BW', 'ROC', 'TOP'): ('TOP', 'ROC'),  
        ('BW', 'STR', 'TOP'): ('TOP', 'STR'),  
        ('BW', 'GS', 'TOP'): ('TOP', 'GS'),    
        }
        
        key = (pred, subj_type, obj_type)
        if key in corrections:
            new_subj_type, new_obj_type = corrections[key]
            corrected = spo.copy()
            corrected['subject']['type'] = new_subj_type
            corrected['object']['type'] = new_obj_type
            return corrected
        
        return None
    
    def _fix_direction_conflicts(self, spo_list: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """修正方向关系冲突 (空间方位互斥性校验)"""
        corrected_spo = []
        issues = []

        direction_relations = ['N', 'S', 'E', 'W', 'EN', 'ES', 'WN', 'WS']
        direction_groups = defaultdict(list)

        for idx, spo in enumerate(spo_list):
            if spo['predicate'] in direction_relations:
                key = (spo['subject']['value'], spo['object']['value'])
                direction_groups[key].append((idx, spo))

        indices_to_remove = set()
        
        for key, relations_with_idx in direction_groups.items():
            if len(relations_with_idx) > 1:

                relations = [r[1] for r in relations_with_idx]
                conflict_found, conflicting_indices = self._check_direction_conflict_details(relations_with_idx)
                
                if conflict_found:

                    first_idx = relations_with_idx[0][0]
                    
                    for idx, spo in relations_with_idx[1:]:      
                         pass 

        processed_pairs = set()
        kept_spos = set() 
        

        remove_flags = [False] * len(spo_list)
        
        for key, relations_with_idx in direction_groups.items():
            if len(relations_with_idx) <= 1:
                continue

            dirs = [(idx, spo['predicate']) for idx, spo in relations_with_idx]
            

            conflict_pairs = [('N', 'S'), ('S', 'N'), ('E', 'W'), ('W', 'E'),
                             ('EN', 'WS'), ('WS', 'EN'), ('ES', 'WN'), ('WN', 'ES')]

            kept_indices_in_group = [relations_with_idx[0][0]]
            
            for i in range(1, len(relations_with_idx)):
                curr_idx, curr_spo = relations_with_idx[i]
                curr_pred = curr_spo['predicate']
                is_conflict = False

                for kept_idx in kept_indices_in_group:
                    kept_pred = spo_list[kept_idx]['predicate']
                    if (curr_pred, kept_pred) in conflict_pairs:
                        is_conflict = True
                        break
                
                if is_conflict:
                    remove_flags[curr_idx] = True
                    issues.append({
                        'type': '方向关系冲突',
                        'relation': curr_spo,
                        'reason': '与已保留的方向关系冲突，已删除'
                    })
                else:
                    kept_indices_in_group.append(curr_idx)

        for i, spo in enumerate(spo_list):
            if not remove_flags[i]:
                corrected_spo.append(spo)
                
        return corrected_spo, issues
    
    def _check_direction_conflict_details(self, relations_with_idx: List[Tuple[int, Dict]]) -> Tuple[bool, List[int]]:

        return False, []

    def _fix_stratigraphic_logic(self, spo_list: List[Dict], text: str = None) -> Tuple[List[Dict], List[Dict]]:

        corrected_spo = []
        issues = []
        

        has_anomaly_keywords = False
        if text:
            for keyword in self.geological_anomaly_keywords:
                if keyword in text:
                    has_anomaly_keywords = True
                    break
        

        strat_ages = {}
        for spo in spo_list:
            if spo['object']['type'] == 'STR':
                age = self._infer_stratigraphic_age(spo['object']['value'])
                if age is not None:
                    strat_ages[spo['object']['value']] = age
            if spo['subject']['type'] == 'STR':
                age = self._infer_stratigraphic_age(spo['subject']['value'])
                if age is not None:
                    strat_ages[spo['subject']['value']] = age
 
        for spo in spo_list:
            if spo['predicate'] == 'IN' and spo['subject']['type'] == 'STR' and spo['object']['type'] == 'STR':
                subject_age = strat_ages.get(spo['subject']['value'])
                object_age = strat_ages.get(spo['object']['value'])
                
                if subject_age and object_age:
                    if subject_age < object_age:
                        if has_anomaly_keywords:
                           
                            corrected_spo.append(spo)
                        else:
                            
                            corrected = {
                                'predicate': 'IN',
                                'subject': spo['object'],
                                'object': spo['subject']
                            }
                            corrected_spo.append(corrected)
                            issues.append({
                                'type': '地层时代顺序错误',
                                'original': spo,
                                'corrected': corrected,
                                'reason': f'{spo["subject"]["value"]}(时代{subject_age})不能包含{spo["object"]["value"]}(时代{object_age})，且文中未提及倒转等异常'
                            })
                    else:
                        corrected_spo.append(spo)
                else:
                    corrected_spo.append(spo)
            else:
                corrected_spo.append(spo)
        
        return corrected_spo, issues
    
    def _infer_stratigraphic_age(self, strat_name: str) -> float:

        for key, age in self.geological_knowledge['stratigraphic_sequence'].items():
            if key in strat_name:
                return age
        
        # 特定地层组判断
        if '组' in strat_name:
            if '马家河' in strat_name or '云梦山' in strat_name or '白草坪' in strat_name:
                return 1.0  
            elif '花峪' in strat_name or '五指岭' in strat_name or '庙坡' in strat_name:
                return 0.5  
            elif '马鞍山' in strat_name:
                return 1.0  
            elif '石千峰' in strat_name:
                return 5.0  
        
        return None
    
    def evaluate_predictions(self, predictions: List[Dict], labels: List[Dict]) -> Dict[str, float]:
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for pred_item, label_item in zip(predictions, labels):
            pred_set = set(
                tuple(
                    sorted(
                        (
                            spo["predicate"],
                            spo["subject"].get("value", "") if isinstance(spo.get("subject", {}).get("value"), list) else spo.get("subject", {}).get("value", ""),
                            spo["subject"].get("type", ""),  
                            spo["object"].get("value", "") if isinstance(spo.get("object", {}).get("value"), list) else spo.get("object", {}).get("value", ""),
                            spo["object"].get("type", ""),  
                        )
                    )
                )
                for spo in pred_item["spo_list"]
                if 'object' in spo and isinstance(spo, dict) and 'subject' in spo and isinstance(spo["subject"], dict)
            )

            label_set = set(
                tuple(
                    sorted(
                        (
                            spo["predicate"],
                            spo["subject"].get("value", "") if isinstance(spo.get("subject", {}).get("value"), list) else spo.get("subject", {}).get("value", ""),
                            spo["subject"].get("type", ""),  # 添加主体的类型
                            spo["object"].get("value", "") if isinstance(spo.get("object", {}).get("value"), list) else spo.get("object", {}).get("value", ""),
                            spo["object"].get("type", ""),  # 添加客体的类型
                        )
                    )
                )
                for spo in label_item["spo_list"]
                if 'object' in spo and isinstance(spo, dict) and 'subject' in spo and isinstance(spo["subject"], dict)
            )

            true_positives += len(pred_set & label_set)
            false_positives += len(pred_set - label_set)
            false_negatives += len(label_set - pred_set)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def process_and_evaluate(self, pred_file: str, label_file: str, output_file: str = None) -> Dict[str, Any]:

        print("=" * 80)
        print("地质关系抽取后处理与评估")
        print("=" * 80)
        print("\n1. 加载数据...")
        predictions = self.load_data(pred_file)
        labels = self.load_text_data(label_file)  
        
        if len(predictions) != len(labels):
            print(f"警告：预测数据({len(predictions)}条)和真实数据({len(labels)}条)数量不一致！")

            min_len = min(len(predictions), len(labels))
            predictions = predictions[:min_len]
            labels = labels[:min_len]
        
        print(f"  加载完成：{len(predictions)}条数据")

        print("\n2. 进行后处理修正...")
        corrected_predictions = []
        all_issues = []
        
        for i, (pred, label) in enumerate(zip(predictions, labels)):
            if 'spo_list' in pred:

                text = label.get('text') if label else None
                

                corrected_spo, issues = self.diagnose_and_correct(pred['spo_list'], text)
                corrected_predictions.append({'spo_list': corrected_spo})
                all_issues.extend(issues)
            else:

                corrected_predictions.append(pred)
        
        print(f"  处理完成：发现{len(all_issues)}个问题")
        

        print("\n3. 评估原始预测结果...")
        original_metrics = self.evaluate_predictions(predictions, labels)

        print("4. 评估后处理后的结果...")
        corrected_metrics = self.evaluate_predictions(corrected_predictions, labels)
        

        print("\n" + "=" * 80)
        print("评估结果汇总")
        print("=" * 80)
        print(f"{'指标':<20} {'原始预测':<15} {'后处理后':<15} {'变化':<10}")
        print("-" * 80)
        print(f"{'精确率(Precision)':<20} {original_metrics['precision']:.4f}{'':<5} {corrected_metrics['precision']:.4f}{'':<5} {corrected_metrics['precision'] - original_metrics['precision']:+.4f}")
        print(f"{'召回率(Recall)':<20} {original_metrics['recall']:.4f}{'':<5} {corrected_metrics['recall']:.4f}{'':<5} {corrected_metrics['recall'] - original_metrics['recall']:+.4f}")
        print(f"{'F1分数':<20} {original_metrics['f1_score']:.4f}{'':<5} {corrected_metrics['f1_score']:.4f}{'':<5} {corrected_metrics['f1_score'] - original_metrics['f1_score']:+.4f}")
        print(f"{'真阳性(TP)':<20} {original_metrics['true_positives']:<15} {corrected_metrics['true_positives']:<15} {corrected_metrics['true_positives'] - original_metrics['true_positives']:+d}")
        print(f"{'假阳性(FP)':<20} {original_metrics['false_positives']:<15} {corrected_metrics['false_positives']:<15} {corrected_metrics['false_positives'] - original_metrics['false_positives']:+d}")
        print(f"{'假阴性(FN)':<20} {original_metrics['false_negatives']:<15} {corrected_metrics['false_negatives']:<15} {corrected_metrics['false_negatives'] - original_metrics['false_negatives']:+d}")
        print("=" * 80)
        

        if all_issues:
            print("\n发现的主要问题类型统计：")
            issue_types = defaultdict(int)
            for issue in all_issues:
                issue_types[issue['type']] += 1
            
            for issue_type, count in sorted(issue_types.items(), key=lambda x: x[1], reverse=True):
                print(f"  {issue_type}: {count}个")
        
        # 7. 保存结果
        if output_file:
            result = {
                'original_metrics': original_metrics,
                'corrected_metrics': corrected_metrics,
                'correction_issues': all_issues,
                'corrected_predictions': corrected_predictions
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n详细结果已保存到: {output_file}")
        
        return {
            'original_metrics': original_metrics,
            'corrected_metrics': corrected_metrics,
            'issues': all_issues,
            'corrected_predictions': corrected_predictions
        }

def main():

    processor = GeologicalRelationProcessor()
    

    pred_file = "xxx"  # 预测结果文件
    label_file = "xxx"      # 真实标签文件
    output_file = "xxx"  # 输出结果文件

    results = processor.process_and_evaluate(pred_file, label_file, output_file)
    

    if results['issues']:
        print("\n实体不在文本中的问题详情（显示前5个）：")
        text_issues = [issue for issue in results['issues'] if issue['type'] == '实体幻觉(无法溯源)' or issue['type'] == '实体幻觉(模糊匹配修正)']
        for i, issue in enumerate(text_issues[:5], 1):
            print(f"\n问题{i}:")
            print(f"  类型: {issue['type']}")
            if 'relation' in issue:
                 print(f"  关系: {issue['relation']['subject']['value']} --{issue['relation']['predicate']}--> {issue['relation']['object']['value']}")
            if 'missing_entities' in issue:
                 print(f"  缺失实体: {', '.join(issue['missing_entities'])}")
            if 'corrections' in issue:
                 print(f"  修正: {', '.join(issue['corrections'])}")
            print(f"  处理: {issue.get('action', '未知')}")

def batch_process(pred_files, label_files, output_dir="results"):
    import os
    processor = GeologicalRelationProcessor()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    all_results = {}
    for i, (pred_file, label_file) in enumerate(zip(pred_files, label_files), 1):
        print(f"\n处理第{i}对文件:")
        print(f"  预测文件: {pred_file}")
        print(f"  标签文件: {label_file}")
        
        output_file = os.path.join(output_dir, f"result_{i}.json")
        results = processor.process_and_evaluate(pred_file, label_file, output_file)
        all_results[f"pair_{i}"] = results
    
    # 汇总结果
    print("\n" + "=" * 80)
    print("批量处理结果汇总")
    print("=" * 80)
    
    return all_results

if __name__ == "__main__": 
    main()
