class KnowledgeBase:
    def __init__(self):
        # To store unique string facts (e.g., "TargetVisible")
        self.facts = set()
        # To store rules as Tuples: ( [list_of_premises], "conclusion_string" )
        self.rules = []

    def tell_fact(self, fact_string):
        self.facts.add(fact_string)

    def tell_rule(self, premise_list, conclusion_string):
        self.rules.append((premise_list, conclusion_string))

    def clear_facts(self):
        self.facts.clear()

    def forward_chain(self):
        new_facts_added = True
        
        while new_facts_added:
            new_facts_added = False
            
            for premises, conclusion in self.rules:
                if conclusion not in self.facts:
                    # Modus Ponens Check: If ALL premises ARE IN facts
                    if all(premise in self.facts for premise in premises):
                        self.facts.add(conclusion)
                        new_facts_added = True