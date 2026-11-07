import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
sns.set_style("whitegrid")
print("Task 3 Started!")

# 1. Pie Chart - Survival
plt.figure(figsize=(6,6))
df['Survived'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#ff6b6b','#4ecdc4'], labels=['Not Survived','Survived'])
plt.title('Overall Survival % - Titanic', fontsize=14, fontweight='bold')
plt.ylabel('')
plt.savefig('1_pie_survival.png')
plt.show()

# 2. Scatter - Age vs Fare
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x='Age', y='Fare', hue='Survived', palette='coolwarm')
plt.title('Age vs Fare by Survival')
plt.savefig('2_scatter_age_fare.png')
plt.show()

# 3. Count - Embarked
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Embarked', hue='Survived', palette='Set2')
plt.title('Survival by Embark Port (S/C/Q)')
plt.savefig('3_embark_survival.png')
plt.show()

# 4. Box - Age by Pclass
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x='Pclass', y='Age', palette='Pastel1')
plt.title('Age Distribution by Passenger Class')
plt.savefig('4_box_age_class.png')
plt.show()

# 5. Bar - Avg Fare by Class
plt.figure(figsize=(6,4))
sns.barplot(data=df, x='Pclass', y='Fare', palette='viridis')
plt.title('Average Fare by Class')
plt.savefig('5_bar_fare_class.png')
plt.show()

# 6. Heatmap - Correlation
plt.figure(figsize=(8,6))
numeric_df = df.select_dtypes(include='number')
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Dashboard - Titanic')
plt.savefig('6_heatmap_correlation.png')
plt.show()

print("Task 3 Completed! 6 Visualizations Saved!")