import pandas as pd
import numpy as np
import random
import os

os.makedirs('data', exist_ok=True)
np.random.seed(42)
random.seed(42)

# Define templates for each category
billing_templates = [
    "I was overcharged on my last invoice by ${amt}. Please refund.",
    "Why is my credit card failing to process?",
    "I want to cancel my subscription and get a refund.",
    "Can you send me the receipt for order #{num}?",
    "There is an unauthorized charge of ${amt} on my account."
]

technical_templates = [
    "The server is down and we cannot access our dashboard! Urgent!",
    "I'm getting a 500 internal server error when trying to upload a file.",
    "The application keeps crashing on my Android phone.",
    "API endpoint /v1/users is returning a timeout.",
    "Your website is loading very slowly today."
]

account_templates = [
    "I forgot my password and the reset link is not working.",
    "How do I change my email address?",
    "My account has been locked. Please unlock it.",
    "I need to add another user to my enterprise plan.",
    "Can you delete my account permanently?"
]

general_templates = [
    "Do you offer a discount for non-profits?",
    "Where is your company located?",
    "What are your business hours?",
    "Is there a free trial available?",
    "I love your product, just wanted to say thanks!"
]

def generate_ticket(category):
    amt = random.randint(10, 500)
    num = random.randint(1000, 9999)
    
    if category == "Billing":
        text = random.choice(billing_templates)
        priority = "High" if "unauthorized" in text or "overcharged" in text else "Medium"
    elif category == "Technical Issue":
        text = random.choice(technical_templates)
        priority = "High" if "server is down" in text or "crashing" in text else "Medium"
    elif category == "Account":
        text = random.choice(account_templates)
        priority = "High" if "locked" in text else ("Medium" if "password" in text else "Low")
    else:  # General Query
        text = random.choice(general_templates)
        priority = "Low"
        
    # Introduce some noise and variations
    text = text.replace("{amt}", str(amt)).replace("{num}", str(num))
    
    # Randomly bump priority or lower it (noise model)
    if random.random() < 0.1:
        priority = random.choice(["High", "Medium", "Low"])
        
    return text, category, priority

def main():
    categories = ["Billing", "Technical Issue", "Account", "General Query"]
    weights = [0.3, 0.4, 0.2, 0.1]  # Technical issues are the most common
    
    data = []
    for i in range(2000):
        cat = random.choices(categories, weights=weights)[0]
        text, true_cat, true_prio = generate_ticket(cat)
        
        # Add random extra words for NLP variety
        prefixes = ["Hi,", "Hello,", "URGENT:", "Please help -", "Just wondering,"]
        if random.random() < 0.5:
            text = random.choice(prefixes) + " " + text
            
        data.append({
            "Ticket_ID": f"TKT-{i+1000}",
            "Text": text,
            "Category": true_cat,
            "Priority": true_prio
        })
        
    df = pd.DataFrame(data)
    
    # Save to CSV
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/synthetic_tickets.csv', index=False)
    print(f"Generated {len(df)} synthetic tickets in data/synthetic_tickets.csv")

if __name__ == "__main__":
    main()
