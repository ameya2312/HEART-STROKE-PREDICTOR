import uuid

import app


def main():
    app.init_db()
    username = f"test_{uuid.uuid4().hex[:8]}"
    created, message = app.create_user("Test User", username, "secret123")
    user = app.authenticate_user(username, "secret123")
    bad_user = app.authenticate_user(username, "wrongpass")

    print(
        {
            "username": username,
            "created": created,
            "message": message,
            "login_ok": bool(user),
            "bad_login_rejected": bad_user is None,
        }
    )


if __name__ == "__main__":
    main()
