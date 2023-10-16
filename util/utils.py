import pandas as pd
import re
from collections import Counter
import ast
import matplotlib.pyplot as plt
import yaml
import os
from pathlib import Path

def load_yaml(path):
    """Load a YAML file from the given path"""
    with open(path, 'r') as file:
        return yaml.safe_load(file)

class KoreanTextAnalysis:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        plt.rcParams['font.family'] = 'D2Coding'
        PROJECT_DIR = Path(__file__).resolve().parents[1]
        data_config_path = os.path.join(PROJECT_DIR, 'config/data_config.yaml')
        data_config = load_yaml(data_config_path)
        self.document = data_config['columns']['document']
        self.preprocessed_document = data_config['columns']['preprocessed_document']
        self.pos = data_config['columns']['pos']
        
    def preprocess_text(self):
        def clean_text(text):
            text = text.lower()
            text = re.sub(r'[^가-힣a-z]', ' ', text)
            tokens = text.split()
            return ' '.join(tokens)
        self.data.dropna(subset=[self.document], inplace=True)
        self.data[self.preprocessed_document] = self.data[self.document].apply(clean_text)
        
    def label_distribution(self):
        label_cols = ['toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        label_counts = self.data[label_cols].sum().sort_values(ascending=False)
        ax = label_counts.plot(kind='bar', color='lightblue', figsize=(20, 10))
        
        for p in ax.patches:
            ax.annotate(
                str(p.get_height()),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 10),
                textcoords='offset points'
                )
    
        plt.title('Distribution of labels')
        plt.ylabel('Number of occurrences', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()
        
    def word_counts(self, n=20, plot=False):
        word_counts = Counter(' '.join(self.data[self.preprocessed_document]).split())
        common_words = word_counts.most_common(20)
        
        if plot:
            words, counts = zip(*common_words)
            plt.figure(figsize=(20, 12))
            plt.bar(words, counts, color='lightblue')
            plt.title(f'Top {n} Most Common Words')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.grid(axis='y')
            plt.show()
            
        return common_words
        
    def pos_tag_distribution(self, column_names):
        self.data[column_names] = self.data[column_names].apply(lambda x: ast.literal_eval(x))
        all_tags = [tag for sublist in self.data[column_names] for _, tag in sublist]
        tag_counts = Counter(all_tags)
        sorted_tag_counts = dict(sorted(tag_counts.items(), key=lambda item: item[1], reverse=True))
        
        return sorted_tag_counts
    