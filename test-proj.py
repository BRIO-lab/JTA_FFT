import open3d as o3d
stl_path = "C:/Users/ajensen123/OneDrive - University of Florida/Documents - Miller Lab/data/Knee/Raw/GMK/GMK_Organized/Patient_2/Session_1/Step_1/GMK_Sphere_right_5_fem.stl"
def main(stl):
    rnd = o3d.visualization.rendering.OffscreenRenderer(1024,1024)
    blk = o3d.visualization.rendering.MaterialRecord()
    blk.base_color = [0.0,0.0,0.0,0.0]
    mesh = o3d.io.read_triangle_mesh(stl)
    mesh.translate([0,0,-1000])
    
    rnd.scene.add_geometry("implant",mesh,blk)
    
    rnd.setup_camera(60,[0,0,0],[0,0,-1],[0,1,0])
    img=rnd.render_to_image()
    o3d.io.write_image("test-proj.png",img,9)
    o3d.visualization.draw_geometries([mesh])

if __name__ == "__main__":
    main(stl_path)