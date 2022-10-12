from composite import composite

src_path = "images/portrait/source.jpg"
trg_path = "images/portrait/target.jpg"
alpha_path = "images/portrait/alpha.jpg"
save_path = "images/portrait/composite.jpg"
w = [0.7, 0.82, 0.84]
composite(src_path, trg_path, alpha_path, w, save_path)
