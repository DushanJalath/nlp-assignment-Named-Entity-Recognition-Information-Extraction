import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.chunk import tree2conlltags
from collections import defaultdict, Counter

# Download required NLTK resources
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)

class EntityExtractor:
    """
    Extracts named entities (Persons, Organizations, Locations)
    Responsibilities: Person 2
    """
    
    def __init__(self):
        self.entities = defaultdict(list)
    
    def extract_entities(self, pos_tags):
        """
        Extract named entities using NLTK's ne_chunk
        """
        # Perform Named Entity Recognition
        chunked = ne_chunk(pos_tags, binary=False)
        
        # Convert tree to IOB tags format
        iob_tags = tree2conlltags(chunked)
        
        # Extract entities
        current_entity = []
        current_label = None
        
        for word, pos, ne_tag in iob_tags:
            if ne_tag.startswith('B-'):  # Beginning of entity
                if current_entity:
                    entity_text = ' '.join(current_entity)
                    self.entities[current_label].append(entity_text)
                current_entity = [word]
                current_label = ne_tag[2:]  # Remove 'B-' prefix
            elif ne_tag.startswith('I-'):  # Inside entity
                current_entity.append(word)
            else:  # Outside entity
                if current_entity:
                    entity_text = ' '.join(current_entity)
                    self.entities[current_label].append(entity_text)
                current_entity = []
                current_label = None
        
        # Add last entity if exists
        if current_entity:
            entity_text = ' '.join(current_entity)
            self.entities[current_label].append(entity_text)
        
        return self.entities
    
    def categorize_entities(self):
        """
        Categorize and count entities
        """
        categorized = {
            'PERSON': self.entities.get('PERSON', []),
            'ORGANIZATION': self.entities.get('ORGANIZATION', []),
            'GPE': self.entities.get('GPE', []),  # Geo-Political Entity (locations)
            'LOCATION': self.entities.get('LOCATION', []),
            'OTHER': []
        }
        
        # Add other entity types to 'OTHER' category
        for entity_type, entity_list in self.entities.items():
            if entity_type not in categorized:
                categorized['OTHER'].extend(entity_list)
        
        return categorized
    
    def get_entity_counts(self):
        """
        Count frequency of each entity
        """
        entity_counts = {}
        for entity_type, entity_list in self.entities.items():
            entity_counts[entity_type] = Counter(entity_list)
        return entity_counts
    
    def display_entities(self):
        """
        Display extracted entities
        """
        categorized = self.categorize_entities()
        
        print("\n" + "="*60)
        print("PERSON 2 OUTPUT: EXTRACTED ENTITIES")
        print("="*60)
        
        for category, entities in categorized.items():
            if entities:
                print(f"\n{category} ({len(entities)} found):")
                # Show unique entities with their counts
                entity_counts = Counter(entities)
                for entity, count in entity_counts.most_common(10):
                    print(f"  - {entity} (appeared {count} time(s))")
        
        return categorized
