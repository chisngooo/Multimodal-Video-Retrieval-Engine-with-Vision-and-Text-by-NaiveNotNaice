def search_frames_with_all_objects(query_objects, data, frame_ids):
    matching_frame_indexes = []
    
    for frame_id in frame_ids:
        frame_data = data.get(frame_id, {})
        frame_objects = set(frame_data.get('objects', []))
        
        # Kiểm tra nếu tất cả các đối tượng trong query_objects có trong frame_objects
        if all(obj.lower() in frame_objects for obj in query_objects):
            matching_frame_indexes.append(frame_id)
    
    return matching_frame_indexes