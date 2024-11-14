import base64
import importlib.metadata
import platform
from sys import stderr
from xml.etree import ElementTree as ET

MODEL_TAG = "Model"
VERSION_TAG = "Version"
TYPE_TAG = "Type"
PARAMETERS_TAG = "Parameters"
PART_TAG = "Part"

def unwrap_model_xml(model_type, xml_data):
    model_data = b""
    for i, xml_datum in enumerate(xml_data, start=1):
        root = ET.fromstring(xml_datum)
        assert root.tag == MODEL_TAG
        assert model_type == root.find(TYPE_TAG).text
        for version in root.findall(VERSION_TAG):
            package = version.get("id")
            if package == "python":
                if version.get("number") != platform.python_version():
                    print(f"Warning: model was trained with python {version.get('number')}, but current Python version is {platform.python_version()}", file=stderr)
            else:
                try:
                    if version.get("number") != importlib.metadata.version(package):
                        print(f"Warning: model was trained with {package} {version.get('number')}, but current {package} version is {importlib.metadata.version(package)}", file=stderr)
                except importlib.metadata.PackageNotFoundError:
                    print(f"Warning: model was trained with {package} {version.get('number')}, but {package} is not installed", file=stderr)
            assert root.find(PART_TAG).get("number") == str(i) and root.find(PART_TAG).get("total") == str(len(xml_data))
        model_data += base64.b64decode(root.find(PARAMETERS_TAG).text)
    return model_data

def wrap_model_xml(model_type, model_data, part_size):
    num_parts = (len(model_data) + part_size - 1) // part_size
    parts = [model_data[i * part_size:(i + 1) * part_size] for i in range(num_parts)]
    xml_data = []
    for i, part in enumerate(parts, start=1):
        root = ET.Element(MODEL_TAG)
        model_type_elem = ET.SubElement(root, TYPE_TAG)
        model_type_elem.text = model_type
        ET.SubElement(root, VERSION_TAG, id="python", number=platform.python_version())
        for dist in importlib.metadata.distributions():
            ET.SubElement(root, VERSION_TAG, id=dist.metadata['Name'], number=dist.version)
        ET.SubElement(root, PART_TAG, number=str(i), total=str(num_parts))
        model_data_elem = ET.SubElement(root, PARAMETERS_TAG)
        model_data_elem.text = base64.b64encode(part).decode("utf-8")
        xml_data.append(ET.tostring(root, encoding="utf-8", method="xml"))
    return xml_data