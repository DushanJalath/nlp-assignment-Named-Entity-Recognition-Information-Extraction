from text_preprocessing_pos_tagging import TextPreprocessor

from EntityExtractor import  EntityExtractor

from relation_extractor import RelationExtractorVisualizer


def main():
    """
    Main function that demonstrates the complete NER system
    """
    
    # Sample text for demonstration
    sample_text = """
    Apple Inc. is an American multinational technology company headquartered in 
    Cupertino, California. Steve Jobs, Steve Wozniak, and Ronald Wayne founded 
    Apple in 1976. Tim Cook currently serves as the CEO of Apple. The company 
    designs, develops, and sells consumer electronics, computer software, and 
    online services. Apple's product line includes the iPhone smartphone, the 
    iPad tablet computer, the Mac personal computer, and the Apple Watch smartwatch.
    
    Microsoft Corporation is another major technology company founded by Bill Gates 
    and Paul Allen in 1975. The company is based in Redmond, Washington. 
    Satya Nadella is the current CEO of Microsoft. Google, founded by Larry Page 
    and Sergey Brin in Mountain View, California, is also a leading technology 
    company.
    
    Amazon, led by Jeff Bezos, revolutionized e-commerce and cloud computing. 
    The company started in Seattle, Washington and has expanded globally. 
    Facebook, now known as Meta, was created by Mark Zuckerberg at Harvard University 
    and is headquartered in Menlo Park, California.
    """
    
    print("="*60)
    print("NER & INFORMATION EXTRACTION SYSTEM")
    print("Group Assignment - 3 Part Integration")
    print("="*60)
    print("\nProcessing text...\n")
    
    # ========== PERSON 1: PREPROCESSING & POS TAGGING ==========
    print("\n🔹 EXECUTING PERSON 1's MODULE...")
    preprocessor = TextPreprocessor()
    preprocessed_data = preprocessor.preprocess_pipeline(sample_text)
    preprocessor.display_pos_stats(preprocessed_data['pos_tags'])
    
    # ========== PERSON 2: ENTITY EXTRACTION ==========
    print("\n🔹 EXECUTING PERSON 2's MODULE...")
    extractor = EntityExtractor()
    entities = extractor.extract_entities(preprocessed_data['pos_tags'])
    categorized_entities = extractor.display_entities()
    
    # ========== PERSON 3: RELATION EXTRACTION & VISUALIZATION ==========
    print("\n🔹 EXECUTING PERSON 3's MODULE...")
    visualizer = RelationExtractorVisualizer(
        preprocessed_data['cleaned_text'],
        entities,
        preprocessed_data['pos_tags']
    )
    
    relations = visualizer.extract_relations()
    visualizer.display_relations()
    visualizer.create_entity_visualization(categorized_entities)
    visualizer.create_wordcloud(categorized_entities, 'PERSON')
    visualizer.create_wordcloud(categorized_entities, 'ORGANIZATION')
    visualizer.generate_report(categorized_entities)
    
    print("\n" + "="*60)
    print("✅ SYSTEM EXECUTION COMPLETED")
    print("="*60)
    print("\nGenerated Files:")
    print("  - entity_distribution.png")
    print("  - person_wordcloud.png")
    print("  - organization_wordcloud.png")
    print("  - ner_report.csv")


if __name__ == "__main__":
    main()
