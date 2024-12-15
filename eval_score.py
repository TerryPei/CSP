import os
import json

def extract_eval_values_html(base_path='outputs', output_md_file='eval_score.md', model_names=None):
    # Initialize a dictionary to hold all results
    results = {}
    datasets = set()

    # Convert model_names to a set for faster lookup; process all if model_names is None
    model_names_set = set(model_names) if model_names else None

    # Walk through the directory structure
    for root, dirs, files in os.walk(base_path):
        # Only proceed if eval.json is found
        if 'eval.json' in files:
            eval_path = os.path.join(root, 'eval.json')
            
            # Extract the method and dataset from the directory structure
            parts = root.split(os.sep)
            if len(parts) >= 3:
                method = parts[-2]
                dataset = parts[-1]
            else:
                continue
            
            # Only process specified models if model_names is not None
            if model_names_set and method not in model_names_set:
                continue
            
            # Add the dataset to the set of datasets
            datasets.add(dataset)
            
            # Read the eval.json file
            with open(eval_path, 'r') as f:
                eval_data = json.load(f)
            
            # Look for either "Accuracy" or "Rouge-L f" in the JSON data
            eval_value = eval_data.get("Accuracy") or eval_data.get("Rouge-L f")
            
            # Store the result in the dictionary
            if method not in results:
                results[method] = {}
            results[method][dataset] = eval_value
    
    # Sort the datasets to ensure consistent table columns
    sorted_datasets = sorted(datasets)
    
    # Generate HTML table header
    html_output = "<table>\n<tr><th>Method</th>"
    for dataset in sorted_datasets:
        html_output += f"<th>{dataset}</th>"
    html_output += "</tr>\n"
    
    # Generate HTML table rows for each method
    for method, dataset_values in results.items():
        html_output += f"<tr><td>{method}</td>"
        for dataset in sorted_datasets:
            # Get the eval result for the dataset, if it exists
            eval_value = dataset_values.get(dataset, "N/A")  # Use "N/A" if no value is found
            html_output += f"<td>{eval_value}</td>"
        html_output += "</tr>\n"
    
    html_output += "</table>"

    # Write the HTML content to a .md file
    with open(output_md_file, 'w') as md_file:
        md_file.write(html_output)
    
    print(f"Markdown file with HTML table '{output_md_file}' has been created.")

# Example usage
# Specify models, or use None to include all
# extract_eval_values_html('outputs', 'eval_score.md', model_names=['new_vlm_cross_softone_topk_only', 'new_vlm_cross_softone_self', 'new_vlm_cross_softone_cross', 'new_vlm_adap', 'new_vlm_softmax_topk_only', 'new_vlm_softone_topk_only', 'new_vlm_softone_cross_only', 'new_vlm_softone_self_only', 'text_prior_pivot_merge_0.1_0.1_speed', 'new_vlm_cross'])
# extract_eval_values_html('outputs', 'eval_score.md', model_names=None)
# extract_eval_values_html('outputs', 'eval_score.md', model_names=[ 'text_prior_pivot_merge_0.1_0.1_speed', 'new_vlm_softone_0.5_sumratiotopk', 'new_vlm_softone_0.9_sumratiotopk', 'new_vlm_softone_0.2_sumratiotopk', 'new_vlm_softone_topk_only', 'new_vlm_softone_0.5', 'new_vlm_softone_0.1_sumratiotopk'])
# extract_eval_values_html('outputs', 'eval_score.md', model_names=['new_vlm_softone_0.9', 'new_vlm_softone_0.1', 'new_vlm_softone_0.5', 'new_csp_v1', 'text_prior_pivot_merge_0.1_0.1_speed'])
extract_eval_values_html('outputs', 'eval_score.md', model_names=['new_csp_v1', 'text_prior_pivot_merge_0.1_0.1_speed'])
