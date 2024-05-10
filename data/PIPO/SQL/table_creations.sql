CREATE TABLE IF NOT EXISTS dimension(
    id INT PRIMARY KEY AUTO_INCREMENT,
    --type ENUM('width', 'length', 'height') NOT NULL,
    d1 FLOAT(4, 2),
    d2 FLOAT(4, 2),
    offset_ FLOAT(4, 2),
    constraint_ INT DEFAULT -1,
    offset_constraint INT DEFAULT -1
    );

CREATE TABLE IF NOT EXISTS angle(
    id INT PRIMARY KEY AUTO_INCREMENT,
    --type ENUM('heading', 'tilt', 'roll') NOT NULL,
    v FLOAT(5, 2),
    constraint_ INT DEFAULT -1,
    );

CREATE TABLE IF NOT EXISTS attach_constraint(
    child INT PRIMARY KEY,
    parent INT DEFAULT -1,
    parent_node TINYINT DEFAULT -1,
    child_node TINYINT DEFAULT -1,
    );

CREATE TABLE IF NOT EXISTS block(
    id INT PRIMARY KEY AUTO_INCREMENT,
    --label VARCHAR(15) NOT NULL,
    --CONSTRAINT chk_label CHECK (label IN ('abutment', 'deck', 'wing_wall', 'haunch', 'edge_beam'),
    label ENUM('abutment', 'deck', 'wing_wall', 'haunch', 'edge_beam'),
    name VARCHAR(20) NOT NULL,
    width INT FOREIGN KEY REFERENCES dimension(id) ON DELETE SET NULL,
    length INT FOREIGN KEY REFERENCES dimension(id) ON DELETE SET NULL,
    height INT FOREIGN KEY REFERENCES dimension(id) ON DELETE SET NULL,
    heading INT FOREIGN KEY REFERENCES angle(id) ON DELETE SET NULL,
    tilt INT FOREIGN KEY REFERENCES angle(id) ON DELETE SET NULL,
    roll INT FOREIGN KEY REFERENCES angle(id) ON DELETE SET NULL,
    );

/* block table is not created before attach_constraint, so I cannot
define the foreign key in the attach_constraint creation */
ALTER TABLE attach_constraint
    ADD FOREIGN KEY(child) REFERENCES block(id) ON DELETE CASCADE
