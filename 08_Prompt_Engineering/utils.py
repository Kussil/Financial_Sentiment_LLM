def process_response(full_response):
    split_response = full_response.split("</s>", 1)
    final_response = split_response[1] if len(split_response) > 1 else split_response[0]
    return final_response
