import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pandas as pd
from collections import Counter
import random
import warnings


# 设置中文字体和样式
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def set_chinese_font():
    """自动设置可用的中文字体"""
    font_path = "C:\Windows\Fonts\simhei.ttf"

    if os.path.exists(font_path):
        # 注册字体
        from matplotlib import font_manager
        font_manager.fontManager.addfont(font_path)
        prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams['font.sans-serif'] = [prop.get_name()]
        plt.rcParams['axes.unicode_minus'] = False
        print(f"✅ 已成功加载中文字体: {prop.get_name()}")
        return

    print("⚠️ 未找到任何中文字体，图表可能无法显示中文！")

class PlantDocAnalyzer:
    def __init__(self, base_path):
        self.base_path = base_path
        self.train_path = os.path.join(base_path, 'Train')
        self.test_path = os.path.join(base_path, 'Test')

    def get_all_categories(self):
        """获取所有类别"""
        if os.path.exists(self.train_path):
            categories = [d for d in os.listdir(self.train_path)
                          if os.path.isdir(os.path.join(self.train_path, d))]
            return sorted(categories)
        return []

    def analyze_category_patterns(self):
        """分析类别命名模式，识别植物和病害类型"""
        categories = self.get_all_categories()

        # 提取植物名称和病害类型
        plants = set()
        diseases = set()
        healthy_plants = set()

        for category in categories:
            # 转换为小写便于处理
            cat_lower = category.lower()

            # 识别健康叶片
            if 'healthy' in cat_lower or 'leaf' in cat_lower and not any(
                    word in cat_lower for word in
                    ['spot', 'rot', 'blight', 'rust', 'mosaic', 'mold', 'mildew', 'virus']):
                plant_name = cat_lower.replace('leaf', '').replace(' ', '').strip()
                if plant_name:
                    healthy_plants.add(plant_name.capitalize())

            # 提取植物名称
            plant_keywords = ['apple', 'pepper', 'blueberry', 'cherry', 'corn', 'grape',
                              'peach', 'potato', 'raspberry', 'soyabean', 'squash', 'strawberry', 'tomato']
            for plant in plant_keywords:
                if plant in cat_lower:
                    plants.add(plant.capitalize())
                    break

            # 提取病害类型
            disease_keywords = {
                'rust': '锈病',
                'scab': '疮痂病',
                'spot': '斑点病',
                'rot': '腐烂病',
                'blight': '枯萎病',
                'mosaic': '花叶病',
                'mold': '霉病',
                'mildew': '霉病',
                'virus': '病毒病',
                'bacterial': '细菌性病害',
                'septoria': '壳针孢病',
                'yellow': '黄化病'
            }

            for eng, chi in disease_keywords.items():
                if eng in cat_lower:
                    diseases.add(chi)
                    break

        return {
            'plants': sorted(list(plants)),
            'diseases': sorted(list(diseases)),
            'healthy_plants': sorted(list(healthy_plants)),
            'total_categories': len(categories)
        }

    def get_detailed_distribution(self, data_path):
        """获取详细的类别分布"""
        if not os.path.exists(data_path):
            return {}

        categories = [d for d in os.listdir(data_path)
                      if os.path.isdir(os.path.join(data_path, d))]

        distribution = {}
        category_details = []

        for category in categories:
            category_path = os.path.join(data_path, category)
            images = [f for f in os.listdir(category_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            num_images = len(images)
            distribution[category] = num_images

            # 获取图片尺寸样本
            sizes = []
            for img_name in images[:5]:  # 只检查前5张
                try:
                    img_path = os.path.join(category_path, img_name)
                    with Image.open(img_path) as img:
                        sizes.append(img.size)
                except:
                    continue

            avg_size = np.mean(sizes, axis=0).astype(int) if sizes else (0, 0)

            category_details.append({
                'category': category,
                'count': num_images,
                'avg_width': avg_size[0],
                'avg_height': avg_size[1]
            })

        return distribution, category_details

    def visualize_plant_disease_analysis(self):
        """可视化植物病害分析"""
        patterns = self.analyze_category_patterns()

        print("=" * 60)
        print("PlantDoc 数据集综合分析")
        print("=" * 60)

        print(f"\n📊 数据集概览:")
        print(f"   总类别数: {patterns['total_categories']}")
        print(f"   涉及植物: {', '.join(patterns['plants'])}")
        print(f"   病害类型: {', '.join(patterns['diseases'])}")
        print(f"   健康叶片类别: {', '.join(patterns['healthy_plants'])}")

        # 训练集分析
        if os.path.exists(self.train_path):
            train_dist, train_details = self.get_detailed_distribution(self.train_path)
            self._create_advanced_visualization(train_dist, train_details, "训练集")

        # 测试集分析
        if os.path.exists(self.test_path):
            test_dist, test_details = self.get_detailed_distribution(self.test_path)
            self._create_advanced_visualization(test_dist, test_details, "测试集")

    def _create_advanced_visualization(self, distribution, details, title):
        """创建高级可视化"""
        if not distribution:
            return

        # 创建子图
        fig = plt.figure(figsize=(20, 15))

        # 1. 类别分布柱状图
        ax1 = plt.subplot(2, 2, 1)
        categories = list(distribution.keys())
        counts = list(distribution.values())

        colors = plt.cm.viridis(np.linspace(0, 1, len(categories)))
        bars = ax1.barh(categories, counts, color=colors)
        ax1.set_title(f'{title} - 类别分布', fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('图片数量', fontsize=12)

        # 在条形上添加数值
        for bar, count in zip(bars, counts):
            width = bar.get_width()
            ax1.text(width + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{count}', ha='left', va='center', fontsize=9)

        # 2. 数据量统计
        ax2 = plt.subplot(2, 2, 2)
        total_images = sum(counts)
        avg_per_class = total_images / len(categories)

        stats_data = {
            '统计指标': ['总图片数', '类别数量', '平均每类图片数', '最多图片类别', '最少图片类别'],
            '数值': [
                total_images,
                len(categories),
                f'{avg_per_class:.1f}',
                f'{categories[np.argmax(counts)]} ({max(counts)})',
                f'{categories[np.argmin(counts)]} ({min(counts)})'
            ]
        }

        ax2.axis('tight')
        ax2.axis('off')
        table = ax2.table(cellText=np.array([stats_data['数值']]).T,
                          rowLabels=stats_data['统计指标'],
                          colLabels=['数值'],
                          cellLoc='center',
                          loc='center',
                          bbox=[0.1, 0.1, 0.8, 0.8])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        ax2.set_title(f'{title} - 统计信息', fontsize=16, fontweight='bold', pad=20)

        # 3. 按植物类型分组统计
        ax3 = plt.subplot(2, 2, 3)
        plant_groups = {}

        for category, count in distribution.items():
            cat_lower = category.lower()
            plant_found = False

            plants = ['apple', 'pepper', 'blueberry', 'cherry', 'corn', 'grape',
                      'peach', 'potato', 'raspberry', 'soyabean', 'squash', 'strawberry', 'tomato']

            for plant in plants:
                if plant in cat_lower:
                    if plant.capitalize() not in plant_groups:
                        plant_groups[plant.capitalize()] = 0
                    plant_groups[plant.capitalize()] += count
                    plant_found = True
                    break

            if not plant_found:
                if '其他' not in plant_groups:
                    plant_groups['其他'] = 0
                plant_groups['其他'] += count

        # 饼图显示植物分布
        wedges, texts, autotexts = ax3.pie(plant_groups.values(),
                                           labels=plant_groups.keys(),
                                           autopct='%1.1f%%',
                                           startangle=90)
        ax3.set_title(f'{title} - 按植物类型分布', fontsize=16, fontweight='bold', pad=20)

        # 4. 健康vs病害分析
        ax4 = plt.subplot(2, 2, 4)
        healthy_count = 0
        disease_count = 0

        for category, count in distribution.items():
            cat_lower = category.lower()
            if 'healthy' in cat_lower or ('leaf' in cat_lower and not any(
                    word in cat_lower for word in
                    ['spot', 'rot', 'blight', 'rust', 'mosaic', 'mold', 'mildew', 'virus', 'bacterial'])):
                healthy_count += count
            else:
                disease_count += count

        health_data = [healthy_count, disease_count]
        health_labels = ['健康叶片', '病害叶片']
        health_colors = ['#2ecc71', '#e74c3c']

        bars = ax4.bar(health_labels, health_data, color=health_colors, alpha=0.8)
        ax4.set_title(f'{title} - 健康 vs 病害', fontsize=16, fontweight='bold', pad=20)
        ax4.set_ylabel('图片数量', fontsize=12)

        # 在柱子上添加数值和百分比
        total_health = sum(health_data)
        for bar, count in zip(bars, health_data):
            height = bar.get_height()
            percentage = (count / total_health) * 100
            ax4.text(bar.get_x() + bar.get_width() / 2., height + max(health_data) * 0.01,
                     f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontsize=12)

        plt.tight_layout()
        plt.show()

        # 打印详细统计
        print(f"\n📈 {title}详细统计:")
        print(f"   总图片数: {total_images}")
        print(f"   健康叶片: {healthy_count} ({healthy_count / total_images * 100:.1f}%)")
        print(f"   病害叶片: {disease_count} ({disease_count / total_images * 100:.1f}%)")
        print(f"   数据不平衡比例: {max(counts) / min(counts):.2f}:1")

    def display_disease_samples(self, num_samples=3):
        """显示各类病害样本图片"""
        if not os.path.exists(self.train_path):
            print("训练集路径不存在")
            return

        categories = self.get_all_categories()

        # 过滤出病害类别（排除健康叶片）
        disease_categories = [cat for cat in categories if 'healthy' not in cat.lower()]

        # 计算布局
        cols = num_samples
        rows = len(disease_categories)

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))

        if rows == 1:
            axes = axes.reshape(1, -1)

        for i, category in enumerate(disease_categories):
            category_path = os.path.join(self.train_path, category)
            images = [f for f in os.listdir(category_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            if len(images) > num_samples:
                sample_images = random.sample(images, num_samples)
            else:
                sample_images = images

            for j, img_name in enumerate(sample_images):
                if j < cols:
                    try:
                        img_path = os.path.join(category_path, img_name)
                        img = Image.open(img_path)

                        # 调整图片大小以便显示
                        img.thumbnail((200, 200))

                        axes[i, j].imshow(img)
                        if j == 0:  # 只在第一列显示类别名称
                            axes[i, j].set_ylabel(category, fontsize=10, rotation=0, ha='right')
                        axes[i, j].set_xticks([])
                        axes[i, j].set_yticks([])

                    except Exception as e:
                        axes[i, j].text(0.5, 0.5, '加载失败',
                                        horizontalalignment='center',
                                        verticalalignment='center',
                                        transform=axes[i, j].transAxes)
                        axes[i, j].set_xticks([])
                        axes[i, j].set_yticks([])

            # 填充空白
            for j in range(len(sample_images), cols):
                axes[i, j].axis('off')

        plt.suptitle('PlantDoc 病害样本展示', fontsize=20, fontweight='bold', y=0.95)
        plt.tight_layout()
        plt.show()


# 使用示例
if __name__ == "__main__":
    # 请修改为您的实际路径
    set_chinese_font()
    dataset_path = "Plantdoc"  # 根据您的描述，应该是这个路径

    if os.path.exists(dataset_path):
        analyzer = PlantDocAnalyzer(dataset_path)
        analyzer.visualize_plant_disease_analysis()
        analyzer.display_disease_samples(num_samples=3)
    else:
        print(f"数据集路径不存在: {dataset_path}")
        print("请检查路径是否正确")