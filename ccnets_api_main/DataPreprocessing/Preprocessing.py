import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler


class DataPreprocessor:
    def __init__(self, df): # df 초기화
        self.df = df
    
    def fillna_zero(self): # 결측치 0으로 마스킹
        self.df.fillna(0, inplace=True)
        return self.df
    
    def dropna_rows(self): # 결측치 존재 행 삭제 
        self.df.dropna(inplace=True)
        return self.df
    
    def one_hot_encode(self): # One-Hot Encoder
        str_columns = self.df.select_dtypes(include=['object']).columns
        self.df = pd.get_dummies(self.df, columns=str_columns)
        return self.df

    def robust_scale(self): # Robust Scaler
        float_columns = self.df.select_dtypes(include=['float']).columns
        scaler = RobustScaler()
        self.df[float_columns] = scaler.fit_transform(self.df[float_columns])
        return self.df
    
    def get_boxplots(self):
        num_cols = [col for col in self.df.columns if self.df[col].dtype in ['int64', 'float64']]
        
        for col in num_cols:
            plt.figure(figsize=(5, 5))
            median_value = self.df[col].median()
            plt.boxplot(self.df[col].fillna(median_value))  # 결측치를 중앙값으로 대체하고 박스플롯을 그림
            plt.title(f'Boxplot of {col}')
            plt.show()
