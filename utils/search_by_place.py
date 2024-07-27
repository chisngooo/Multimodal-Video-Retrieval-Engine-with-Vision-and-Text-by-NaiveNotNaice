def search_frames_with_any_place(query_place, data):
    matching_frame_indexes = []
    
    for frame_id, frame_data in data.items():
        frame_place = frame_data.get('place', "").lower()
        
        # Kiểm tra nếu query_place có xuất hiện trong frame_place
        if query_place.lower() in frame_place:
            matching_frame_indexes.append(frame_data['id'])
    
    return matching_frame_indexes