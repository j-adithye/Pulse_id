"""
cli.py - Terminal interface for palm vein auth (1:1 verification)
Runs as a continuous loop until q or Ctrl+C.

Usage:
    python3 cli.py
"""

import sys
from camera import capture, close, get_camera
from auth import register, verify
from embeddings import list_users, delete_user


def prompt_capture(label):
    input("  Position hand (" + label + ") and press Enter to capture...")
    img = capture()
    print("  Captured.")
    return img


def cmd_register():
    print("=" * 50)
    print("REGISTER")
    print("=" * 50)
    username = input("Enter username: ").strip()
    if not username:
        print("[ERROR] Username cannot be empty")
        return
    print("Capturing 3 images for: " + username)
    print("Place hand in LEFT, CENTRE, RIGHT positions when prompted.")
    images = []
    for pos in ["LEFT", "CENTRE"]:
        img = prompt_capture(pos)
        images.append(img)
    print("Processing...")
    result = register(username, images)
    if result["success"]:
        print("[SUCCESS] " + result["message"])
    else:
        print("[FAILED]  " + result["message"])


def cmd_login():
    print("=" * 50)
    print("LOGIN")
    print("=" * 50)
    username = input("Enter username: ").strip()
    if not username:
        print("[ERROR] Username cannot be empty")
        return
    img = prompt_capture("scan position")
    print("Processing...")
    result = verify(username, img)
    if result["success"]:
        dist = result.get("distance")
        matched = result.get("matched", "")
        extra = ""
        if dist is not None:
            extra += "  distance=" + str(round(dist, 4))
        if matched:
            extra += "  matched=" + matched
        print("[SUCCESS] " + result["message"] + extra)
    else:
        print("[FAILED]  " + result["message"])


def cmd_list():
    users = list_users()
    if not users:
        print("No users registered.")
        return
    print(str(len(users)) + " registered user(s):")
    for u in users:
        print("  - " + u)


def cmd_delete():
    cmd_list()
    username = input("Enter username to delete: ").strip()
    if not username:
        print("[ERROR] Username cannot be empty")
        return
    confirm = input("Delete \"" + username + "\"? (y/n): ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return
    result = delete_user(username)
    if result["success"]:
        print("[SUCCESS] " + result["message"])
    else:
        print("[FAILED]  " + result["message"])


def main():
    print("=" * 50)
    print("DORSAL VEIN AUTH")
    print("Ctrl+C to exit")
    print("=" * 50)

    # Initialize camera once at startup
    get_camera()

    try:
        while True:
            print("")
            print("1. Register")
            print("2. Login")
            print("3. List users")
            print("4. Delete user")
            print("5. Exit")
            choice = input("Enter choice: ").strip()

            if choice == "1":
                cmd_register()
            elif choice == "2":
                cmd_login()
            elif choice == "3":
                cmd_list()
            elif choice == "4":
                cmd_delete()
            elif choice == "5" or choice.lower() == "q":
                break
            else:
                print("[ERROR] Invalid choice")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        close()


if __name__ == "__main__":
    main()