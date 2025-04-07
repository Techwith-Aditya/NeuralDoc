import json

class LLMPrompt:
    def __init__(self):
        self.user_results = []

    def _save_prompt(self, content, filename):
        """Save prompt content to file"""
        with open(f'./extracted/{filename}', 'w', encoding='utf-8') as f:
            f.write(content)

    def prompt_for_user_based_search(self, search_result):
        """Generate main content summary prompt"""
        self.user_results = [entry for v in search_result["user_based_search"].values() for entry in v]
        user_input = json.dumps(self.user_results, indent=4)

        prompt = f'''    
        Generate concise, structured headings and summaries from these query matches (similarity ≥0.85):
        - Extract headings from "sub_heading" fields (remove leading numbers)
        - Deduplicate similar headings
        - Maintain logical flow and key insights
        - Synthesize different perspectives
        
        Input: {user_input}
        
        Format:
        ## [Cleaned Heading]
        [Concise summary text]
        '''
        self._save_prompt(prompt, 'userbased.txt')
        return prompt

    def _section_prompt(self, section_name, search_result, default_key):
        """Generate section prompt template"""
        content = search_result.get("default_results", {}).get(section_name) or search_result["user_based_search"].items()
        return f'''
        Generate academic {section_name.lower()} summary from these sources:
        - Formal, academic tone
        - Highlight key elements and common themes
        - Remove redundancy while maintaining coherence
        - Include significant findings/methods
        
        Input: {json.dumps({section_name: content}, indent=4)}
        '''

    def prompt_for_intro(self, search_result):
        prompt = self._section_prompt("Introduction", search_result)
        self._save_prompt(prompt, 'introduction.txt')
        return prompt

    def prompt_for_abstract(self, search_result):
        prompt = self._section_prompt("Abstract", search_result)
        self._save_prompt(prompt, 'abstract.txt')
        return prompt

    def prompt_for_conclusion(self, search_result):
        prompt = self._section_prompt("Conclusion", search_result)
        self._save_prompt(prompt, 'conclusion.txt')
        return prompt

    def prompt_for_reference(self, search_result):
        """Generate formatted reference list prompt"""
        refs = search_result.get("default_results", {}).get("References") or search_result["user_based_search"].items()
        input_data = json.dumps({"References": refs}, indent=4)

        prompt = f'''
        Format these references into IEEE style:
        - Deduplicate entries
        - Maintain consistent formatting
        - Remove irrelevant papers
        - Serialize with [N] numbering
        
        Example:
        [1] Author. "Title", Journal, vol, pp, year.
        
        Input: {input_data}
        '''
        self._save_prompt(prompt, 'reference.txt')
        return prompt

    def prompt_for_methodology(self, search_result):
        prompt = self._section_prompt("Methodology", search_result)
        self._save_prompt(prompt, 'methodology.txt')
        return prompt

    def prompt_for_result(self, search_result):
        prompt = self._section_prompt("Results", search_result)
        self._save_prompt(prompt, 'results.txt')
        return prompt

    def prompt_for_lit_review(self, references):
        """Generate literature review prompt"""
        prompt = f'''
        Create IEEE-style literature review from these references:
        - Group related studies
        - Use single citations: [1] not [
