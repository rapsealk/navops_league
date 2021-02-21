CREATE TABLE rimpac.model (
	id CHAR(20) DEFAULT 'abc123' NOT NULL PRIMARY KEY,
	rating INT UNSIGNED DEFAULT 1200 NOT NULL,
	`path` varchar(100)
)
ENGINE=InnoDB
DEFAULT CHARSET=utf8mb4
COLLATE=utf8mb4_general_ci;

CREATE TABLE rimpac.match_history (
	id INTEGER UNSIGNED auto_increment NOT NULL PRIMARY KEY,
	home CHAR(20) NOT NULL,
	away CHAR(20) NOT NULL,
	`result` INTEGER UNSIGNED DEFAULT 0 NOT NULL,
	`timestamp` DATETIME NOT NULL,
	CONSTRAINT FOREIGN KEY (home) REFERENCES rimpac.model (id)
		ON DELETE CASCADE
		ON UPDATE RESTRICT,
	CONSTRAINT FOREIGN KEY (away) REFERENCES rimpac.model (id)
		ON DELETE CASCADE
		ON UPDATE RESTRICT
)
ENGINE=InnoDB
DEFAULT CHARSET=utf8mb4
COLLATE=utf8mb4_general_ci;