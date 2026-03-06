import xml.etree.ElementTree as ET

def fix_urdf_and_offset_zeros(input_file, output_file):
    # Offsets for the new "zero" position
    q1_offset = 0.5772411021609789
    q2_offset = -1.0307324155578172

    tree = ET.parse(input_file)
    root = tree.getroot()

    # 1. Fix baseHinge (Connection between base and arm1)
    base_hinge = root.find(".//joint[@name='baseHinge']")
    # Create origin if missing
    origin1 = base_hinge.find('origin')
    if origin1 is None:
        origin1 = ET.SubElement(base_hinge, 'origin')
    origin1.set('rpy', f"0 0 {q1_offset}")
    origin1.set('xyz', "0 0 0")

    # 2. Fix interArm (Connection between arm1 and arm2)
    inter_arm = root.find(".//joint[@name='interArm']")
    origin2 = inter_arm.find('origin')
    # This must stay at xyz="0.75 0 0" but add the RPY offset
    origin2.set('rpy', f"0 0 {q2_offset}")
    origin2.set('xyz', "0.75 0 0")

    # 3. CRITICAL: Reset Link Visuals/Collision to 0
    # Because the JOINT now handles the rotation, the BOX shouldn't have its own RPY
    for link_name in ['arm1', 'arm2']:
        link = root.find(f".//link[@name='{link_name}']")
        for tag in ['visual', 'collision', 'inertial']:
            elem = link.find(tag)
            if elem is not None:
                origin = elem.find('origin')
                if origin is not None:
                    # Keep xyz (to center the box), but reset rpy to 0
                    current_xyz = origin.get('xyz', '0 0 0')
                    origin.set('rpy', '0 0 0')
                    origin.set('xyz', current_xyz)

    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    print(f"Fixed URDF saved as {output_file}")

fix_urdf_and_offset_zeros('/home/maryammahmood/xdaadbot_ws/src/daadbot_desc/urdf/2_link_urdf/2link_robot.urdf.xacro', '/home/maryammahmood/xdaadbot_ws/src/daadbot_desc/urdf/2_link_urdf/2link_robot_fixed.urdf.xacro')