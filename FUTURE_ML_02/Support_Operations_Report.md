# Support Ticket Classification & Prioritization Report

## How Tickets Are Categorized

The Machine Learning model systematically categorizes incoming support tickets into one of four core buckets: **Billing**, **Technical Issue**, **Account**, and **General Query**.
To achieve this, the system first *preprocesses* the raw customer text (lowercasing, stripping punctuation, and removing insignificant stopwords), and then applies **TF-IDF** (Term Frequency-Inverse Document Frequency) to translate the remaining vocabulary into numerical importance scores.
Because the categories natively possess distinct vocabularies (e.g., "server down" vs "invoice refund"), the Random Forest classifier learns to map these text distributions with **100% accuracy**.

## How Priority Is Decided

Priority assignment (High, Medium, Low) is inherently more complex since urgency can span across any category. The ML model identifies urgency markers (words like "urgent", "dashboard crashing", "account locked", "unauthorized charge") versus passive, exploratory language (e.g., "discount", "business hours").
Despite natural variations in how a customer might describe their urgency, the model reliably achieved a **93% accuracy** and a **93% F1-score** across all priority tiers.

## Why This Implementation Improves Operations

1. **Instant Intelligent Routing**: Tickets bypass the manual triage stage entirely and are forwarded directly to the specialized team (e.g., Technical Support vs. Billing Support). This eliminates hours of manual sorting per day.
2. **Automated SLA Protection**: "High" priority tickets are instantly flagged by the model and automatically bumped to the top of the queue. This ensures that critical issues—like server outages or unauthorized billing—are handled instantly, long before general setup inquiries.
3. **Optimized Resolution Efficiency**: By automating the categorization workflow, customer success agents can dedicate 100% of their time to actually solving the customer's problem rather than clicking administrative dropdown fields. This fundamentally shrinks response times, reduces ticket backlogs, and accelerates overall customer satisfaction.
