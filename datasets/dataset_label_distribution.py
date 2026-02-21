import pandas as pd
import matplotlib.pyplot as plt

# Load data
train = pd.read_csv('train_dataset.csv')
dev = pd.read_csv('test_dataset.csv')

# Color palette
colors_palette = ['#5B9BD5', '#2E5C8A', '#F4B183', '#D4A574', '#70AD47', '#548235', '#9E7BB5', '#6F4E8C', '#7F7F7F']

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Top-left: Level 1 Train
train['clarity_label'].value_counts().plot(kind='bar', ax=axes[0, 0], color=colors_palette[:3])
axes[0, 0].set_title('Clarity Distribution - Train')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_xlabel('')
axes[0, 0].tick_params(axis='x', rotation=90)

# Top-right: Level 1 Dev
dev['clarity_label'].value_counts().plot(kind='bar', ax=axes[0, 1], color=colors_palette[:3])
axes[0, 1].set_title('Clarity Distribution - Dev')
axes[0, 1].set_ylabel('')
axes[0, 1].set_xlabel('')
axes[0, 1].tick_params(axis='x', rotation=90)

# Bottom-left: Level 2 Train
train_evasion_order = train['evasion_label'].value_counts().index
train['evasion_label'].value_counts().reindex(train_evasion_order).plot(kind='bar', ax=axes[1, 0], color=colors_palette)
axes[1, 0].set_title('Evasion Distribution - Train')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_xlabel('')
axes[1, 0].tick_params(axis='x', rotation=90)

# Bottom-right: Level 2 Dev (3 annotators, ordered by train)
labels = train_evasion_order
x = range(len(labels))
width = 0.25

ann1 = [len(dev[dev['annotator1'] == l]) for l in labels]
ann2 = [len(dev[dev['annotator2'] == l]) for l in labels]
ann3 = [len(dev[dev['annotator3'] == l]) for l in labels]

axes[1, 1].bar([i - width for i in x], ann1, width, label='Annotator 1', color='#5B9BD5')
axes[1, 1].bar(x, ann2, width, label='Annotator 2', color='#70AD47')
axes[1, 1].bar([i + width for i in x], ann3, width, label='Annotator 3', color='#9E7BB5')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(labels, rotation=90, ha='right')
axes[1, 1].set_title('Evasion Distribution - Dev')
axes[1, 1].set_ylabel('')
axes[1, 1].set_xlabel('')
axes[1, 1].legend()

axes[0, 0].tick_params(axis='x', labelsize=20)
axes[0, 1].tick_params(axis='x', labelsize=20)
axes[1, 0].tick_params(axis='x', labelsize=18)
axes[1, 1].tick_params(axis='x', labelsize=18)

for ax in axes.flat:
    ax.tick_params(axis='y', labelsize=11)

axes[0, 0].title.set_fontsize(20)
axes[0, 1].title.set_fontsize(20)
axes[1, 0].title.set_fontsize(20)
axes[1, 1].title.set_fontsize(20)

plt.tight_layout()
plt.savefig('dataset_distributions.png', dpi=300, bbox_inches='tight')
plt.show()