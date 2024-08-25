import json

# Đọc dữ liệu từ file JSON
input_file = 'OBJECT.json'  # Thay đổi tên file nếu cần
output_file = 'output.json'  # Tên file để lưu kết quả

# Hàm chuyển đổi chuỗi thành định dạng mong muốn
def convert_format(entry):
    parts = entry.split()
    quantity = int(parts[0])
    object_type = parts[1]
    color = parts[2]
    return [quantity, object_type, color]

# Đọc dữ liệu từ file
with open(input_file, 'r') as file:
    data = json.load(file)

# Chuyển đổi dữ liệu
converted_data = {}
for key, values in data.items():
    converted_data[key] = [convert_format(value) for value in values]

# Lưu kết quả vào file JSON
with open(output_file, 'w') as file:
    json.dump(converted_data, file, indent=4)

print(f'Kết quả đã được lưu vào {output_file}')
