template_rev_2 = """<s> Given the text from a financial news article, analyze the content and produce a structured summary categorizing the company's performance and related impacts across multiple predefined categories. Each category should be listed with a corresponding sentiment derived from the article. If a category is not mentioned or relevant based on the article content, mark it as 'N/A'. Ensure that all categories are addressed for a comprehensive summary.

Categories and Sentiment Options to be Analyzed:
• Financial Targets
  • Greatly Exceeded Target
  • Exceeded Target
  • Achieved Target
  • Below Target
  • Significantly below Target
• Exploration / Discoveries
  • Major Discovery
  • Minor Discovery
  • Favorable Exploration Results
  • Unfavorable Exploration Results
  • Exploration Failure
• Reserves
  • Significant Reserves Add
  • Minor Reserves Add
  • Stable Reserve Levels
  • Small Reserves Loss / Writeoff
  • Significant Reserves Depletions / Writeoff
• Production Targets
  • Greatly Exceeded Target
  • Exceeded Target
  • Achieved Target
  • Below Target
  • Significantly below Target
• New Energy Investments / Projects
  • Major Advancements in New Energy Initiatives
  • Minor Advancements in New Energy Initiatives
  • Setback in New Energies Project
  • New Energy Projects Abandoned or Failed
• Acquisitions and Mergers
  • Major Acquisition or Merger
  • Minor Acquisition or Merger
  • Delay in Acquisition or Merger
  • Cancelled Acquisition or Merger
• Divestments
  • Major Divestment
  • Small Divestment
• Public Sentiment
  • Very Positive
  • Positive
  • Neutral
  • Negative
  • Very Negative
• Regulatory / Geopolitical Factors
  • Favorable change to Operations
  • Potential Large Disruption to Operations
  • Potential Small Disruption to Operations
• Environmental Factors
  • Very Positive
  • Positive
  • Neutral
  • Negative
  • Very Negative

Output Format:
• Use bullet points for each category.
• For each category, provide the status or sentiment extracted from the article. If no specific information is available, indicate 'N/A'.

Example Output:
• Financial Targets - Achieved Target
• Exploration / Discoveries - N/A
• Reserves - Minor Reserves Add
• Production Targets - Exceeded Target
• New Energy Investments / Projects - Major Advancements in New Energy Initiatives
• Acquisitions and Mergers - Major Acquisition or Merger
• Divestments - N/A
• Public Sentiment - Very Positive
• Regulatory / Geopolitical Factors - Potential Small Disruption to Operations
• Environmental Factors - Neutral

**Additional Examples**:
• Financial Targets - Below Target
• Exploration / Discoveries - Major Discovery
• Reserves - Significant Reserves Depletions / Writeoff
• Production Targets - Below Target
• New Energy Investments / Projects - Minor Advancements in New Energy Initiatives
• Acquisitions and Mergers - Delay in Acquisition or Merger
• Divestments - Small Divestment
• Public Sentiment - Negative
• Regulatory / Geopolitical Factors - Favorable change to Operations
• Environmental Factors - Very Positive

**Steps to Ensure Accuracy**:
1. **Contextual Reinforcement**: Re-read the article text to ensure you fully understand the context.
2. **Iterative Refinement**: After generating the initial summary, review each category and verify accuracy. If unsure, re-evaluate the specific sections of the article.
3. **Ambiguity Clarification**: If any category is ambiguous, provide a brief justification for the chosen sentiment.
4. **Self-Verification**: Once the summary is complete, re-read it to ensure coherence and accuracy.

**Constraints**: ONLY RESPOND USING THE PROVIDED FORMAT/n/n

The text from the financial news article is below:
{article}</s>


"""