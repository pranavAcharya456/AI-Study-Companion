import argparse
import getpass
import os
import sys

from backend import AuthService, Repo


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create-admin"); c.add_argument("--org", required=True); c.add_argument("--id", required=True); c.add_argument("--email", required=True)
    a = ap.parse_args(argv)
    secret = os.environ.get("APP_SECRET", "").encode()
    if len(secret) < 32:
        print("APP_SECRET must be set (32+ bytes).", file=sys.stderr); return 2
    pw = getpass.getpass("Password: ")
    if pw != getpass.getpass("Repeat password: "):
        print("Passwords differ.", file=sys.stderr); return 2
    repo = Repo(os.environ.get("APP_DB", "sqlite+pysqlite:///study_companion.db"))
    try:
        repo.ensure_org(a.org)
        AuthService(repo, secret).create_user(a.org, a.id, a.email, pw, "org_admin")
    except Exception as e:
        print(f"Could not create admin: {e}", file=sys.stderr); return 1
    print(f"Created org admin '{a.id}' for org '{a.org}'."); return 0


if __name__ == "__main__":
    raise SystemExit(main())
