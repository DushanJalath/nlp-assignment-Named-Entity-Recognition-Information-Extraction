import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')
nltk.download('maxent_ne_chunker_tab')
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.chunk import tree2conlltags
from nltk.corpus import stopwords
import re
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd

class RelationExtractorVisualizer:

    def __init__(self, text, entities, pos_tags):
        self.text = text
        self.entities = entities
        self.pos_tags = pos_tags
        self.relations = []

    def extract_relations(self):
        sentences = nltk.sent_tokenize(self.text)

        # Common relation patterns
        relation_verbs = ['works', 'founded', 'created', 'visited', 'lives',
                         'manages', 'leads', 'joined', 'met', 'from']

        for sentence in sentences:
            tokens = word_tokenize(sentence)

            # Find entities in this sentence
            sentence_entities = []
            for entity_type, entity_list in self.entities.items():
                for entity in entity_list:
                    if entity in sentence:
                        sentence_entities.append((entity, entity_type))

            # Find relations between entities
            for i, (entity1, type1) in enumerate(sentence_entities):
                for entity2, type2 in sentence_entities[i+1:]:
                    # Find words between entities
                    try:
                        idx1 = sentence.index(entity1)
                        idx2 = sentence.index(entity2)
                        between = sentence[idx1+len(entity1):idx2]

                        # Check for relation verbs
                        for verb in relation_verbs:
                            if verb in between.lower():
                                relation = {
                                    'entity1': entity1,
                                    'type1': type1,
                                    'relation': verb,
                                    'entity2': entity2,
                                    'type2': type2,
                                    'sentence': sentence
                                }
                                self.relations.append(relation)
                    except:
                        continue

        return self.relations

    def create_entity_visualization(self, categorized_entities):
        """
        Create bar chart of entity counts
        """
        plt.figure(figsize=(12, 6))

        # Count entities by category
        categories = []
        counts = []

        for category, entities in categorized_entities.items():
            if entities:
                categories.append(category)
                counts.append(len(entities))

        plt.bar(categories, counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'])
        plt.xlabel('Entity Type', fontsize=12, fontweight='bold')
        plt.ylabel('Count', fontsize=12, fontweight='bold')
        plt.title('Named Entity Distribution', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('entity_distribution.png', dpi=300, bbox_inches='tight')
        print("\n📊 Entity distribution chart saved as 'entity_distribution.png'")
        plt.close()

    def create_wordcloud(self, categorized_entities, entity_type='PERSON'):
        """
        Create word cloud for specific entity type
        """
        entities = categorized_entities.get(entity_type, [])

        if entities:
            entity_text = ' '.join(entities)
            wordcloud = WordCloud(width=800, height=400,
                                background_color='white',
                                colormap='viridis').generate(entity_text)

            plt.figure(figsize=(12, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'{entity_type} Word Cloud', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{entity_type.lower()}_wordcloud.png', dpi=300, bbox_inches='tight')
            print(f"☁️  {entity_type} word cloud saved as '{entity_type.lower()}_wordcloud.png'")
            plt.close()

    def display_relations(self):
        """
        Display extracted relations
        """
        print("\n" + "="*60)
        print("PERSON 3 OUTPUT: EXTRACTED RELATIONS & VISUALIZATIONS")
        print("="*60)

        if self.relations:
            print(f"\nFound {len(self.relations)} relations:")
            for i, rel in enumerate(self.relations[:10], 1):
                print(f"\n{i}. {rel['entity1']} ({rel['type1']}) "
                      f"--[{rel['relation']}]--> "
                      f"{rel['entity2']} ({rel['type2']})")
                print(f"   Context: \"{rel['sentence'][:100]}...\"")
        else:
            print("\nNo explicit relations found in the text.")

    def generate_report(self, categorized_entities):
        """
        Generate a comprehensive report
        """
        report_data = []

        for category, entities in categorized_entities.items():
            if entities:
                entity_counts = Counter(entities)
                for entity, count in entity_counts.items():
                    report_data.append({
                        'Entity Type': category,
                        'Entity': entity,
                        'Frequency': count
                    })

        df = pd.DataFrame(report_data)
        df = df.sort_values(['Entity Type', 'Frequency'], ascending=[True, False])

        print("\n" + "="*60)
        print("COMPREHENSIVE ENTITY REPORT")
        print("="*60)
        print(df.to_string(index=False))

        # Save to CSV
        df.to_csv('ner_report.csv', index=False)
        print("\n📄 Detailed report saved as 'ner_report.csv'")

        return df
