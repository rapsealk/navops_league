CREATE TABLE rimpac.leaguer (
	id INT UNSIGNED auto_increment NOT NULL PRIMARY KEY,
	rating INT UNSIGNED DEFAULT 1200 NOT NULL
)
ENGINE=InnoDB
DEFAULT CHARSET=utf8mb4
COLLATE=utf8mb4_general_ci
COMMENT='League participants.';