import pandas as pd
import json
import os

# Define file paths
BASE_DIR = "."  # Points to the current directory
JSON_BIZ = "datasets/yelp_json/yelp_academic_dataset_business.json"
JSON_REV = "datasets/yelp_json/yelp_academic_dataset_review.json"
OUTPUT_DIR = "warp_output"

def prepare_yelp_for_warprec():
    # Create the output folder if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Folder created: {OUTPUT_DIR}")

    print("Loading Business (Context)...")
    biz_data = []
    with open(JSON_BIZ, 'r', encoding='utf-8') as f:
        for line in f:
            b = json.loads(line)
            # Extract the main category and city as context features
            main_cat = b['categories'].split(',')[0] if b['categories'] else 'None'
            biz_data.append({
                'item_id': b['business_id'],
                'context_cat': main_cat,
                'context_city': b['city']
            })
    df_biz = pd.DataFrame(biz_data)

    print("Loading Review (Interactions)...")
    rev_data = []
    # Use a limit (e.g., 200k rows) to avoid running out of RAM if necessary
    with open(JSON_REV, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            rev_data.append({
                'user_id': r['user_id'],
                'item_id': r['business_id'],
                'rating': r['stars'],
                'timestamp': r['date']
            })
            if i >= 200000: break 

    df_rev = pd.DataFrame(rev_data)

    # Merge and encode IDs (WarpRec works better with integer IDs)
    print("Merging and Label Encoding...")
    final_df = pd.merge(df_rev, df_biz, on='item_id')
    
    for col in ['user_id', 'item_id', 'context_cat', 'context_city']:
        final_df[col] = pd.Categorical(final_df[col]).codes

    # Final save
    output_path = os.path.join(OUTPUT_DIR, "yelp_warprec_ready.tsv")
    final_df.to_csv(output_path, sep='\t', index=False)
    print(f"Conversion completed! File saved to: {output_path}")

if __name__ == "__main__":
    prepare_yelp_for_warprec()