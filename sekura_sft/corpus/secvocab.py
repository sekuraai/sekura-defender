"""Controlled security vocabulary + entity regexes.

Used two ways: to score document-level security purity (docset.py) and to reject
zero-information claims (filters.py). Kept deliberately broad on the tag side --
it is a *proposal* generator for human review, not an unattended gate.
"""

from __future__ import annotations

import re

SECURITY_TAGS = frozenset("""
security vulnerability vulnerabilities exploit exploitation attack attacker owasp cwe cve
xss sqli sql-injection csrf ssrf rce lfi rfi xxe ssti idor deserialization
authentication authorization authn authz session cookie jwt oauth saml mfa rbac zero-trust
crypto cryptography tls ssl encryption decryption hashing signing certificate pki pqc
password credential credentials secret secrets keys keystore vault
firewall waf ids ips siem soc edr xdr detection logging telemetry sigma yara snort suricata
incident incident-response forensics dfir malware ransomware phishing trojan rootkit backdoor
pentest pentesting redteam blueteam purpleteam recon reconnaissance enumeration scanning
privilege-escalation privesc lateral-movement persistence exfiltration c2 command-and-control
nmap metasploit burp wireshark kali nessus nikto sqlmap hydra john hashcat mimikatz bloodhound
gobuster ffuf amass subfinder nuclei masscan responder impacket crackmapexec bettercap aircrack
semgrep bandit trivy gitleaks checkov gosec brakeman njsscan snyk osv
threat threat-modeling threat-intel risk compliance audit hardening baseline cis-benchmark
nist iso27001 pci-dss hipaa gdpr soc2 fedramp asvs wstg mastg samm
mitre att&ck attack-pattern tactic technique capec ioc iocs ttp
kerberos ldap active-directory ntlm smb rdp winrm dcsync kerberoasting adcs golden-ticket
container kubernetes k8s docker podman cloud-security iam s3 ec2 imds metadata-service
aws azure gcp cspm cnapp cwpp serverless lambda
reverse-engineering binary-exploitation buffer-overflow rop shellcode heap stack-canary aslr
fuzzing fuzzer symbolic-execution taint-analysis sast dast sca iast
network-security packet-capture arp dns-security email-security spf dkim dmarc vpn ipsec
wifi wireless bluetooth rfid nfc sdr
supply-chain sbom dependency-confusion typosquatting cicd pipeline
prompt-injection jailbreak model-extraction ai-security llm-security ai-redteam
sanitize sanitized sanitization sanitizer escape escaping encode encoding decoding
validate validation input-validation output-encoding allowlist whitelist denylist blocklist
throttle throttling rate-limit rate-limiting backoff lockout captcha proof-of-work
permission permissions privilege least-privilege access-control acl entitlement
untrusted trusted trust-boundary tainted sink source dataflow
csp cors sameorigin samesite httponly secure-flag hsts referrer-policy subresource-integrity
csrf-token nonce salt iv kdf pbkdf2 bcrypt scrypt argon2 hmac mac aead gcm cbc
session-timeout idle-timeout re-authentication step-up passwordless webauthn fido2 totp
audit-log logging monitoring alerting anomaly baseline integrity availability confidentiality
patch patching update upgrade version-pinning dependency-pinning lockfile
webroot chroot sandbox seccomp apparmor selinux capabilities namespace cgroup
secure-by-default defense-in-depth fail-secure fail-closed principle-of-least-privilege
misconfiguration hardening remediation mitigation countermeasure control safeguard
penetration payload payloads injection bypass overflow escalation enumerate harden
pcap tcpdump packet-sniffing debugger disassembler decompiler gdb objdump strace ltrace
tamper spoofing spoof hijack hijacking replay-attack mitm man-in-the-middle
blackbox whitebox greybox black-box white-box grey-box scope engagement rules-of-engagement
""".split())

# Language gate. The corpus carries full ASVS translations (Japanese, French,
# German, Spanish, ...). They are legitimate security content but this is an
# English model, so they are dropped with their own reason code rather than
# being hidden inside a generic low-information bucket.
RE_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
RE_LATIN = re.compile(r"[A-Za-z]")


def is_probably_english(text: str) -> bool:
    """Cheap, dependency-free language gate. CJK script is decisive; for Latin
    scripts, require a floor of common English function words."""
    if not text:
        return True
    if RE_CJK.search(text):
        return False
    letters = RE_LATIN.findall(text)
    if len(letters) < 20:
        return True                      # too short to judge; let other gates decide
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if len(words) < 6:
        return True
    stop = {"the", "of", "and", "to", "a", "in", "is", "for", "that", "be", "are",
            "or", "with", "as", "this", "must", "should", "can", "if", "on", "by",
            "an", "from", "not", "it", "which", "when", "all", "any", "has", "have"}
    return sum(1 for w in words if w in stop) / len(words) >= 0.10

# Entity patterns. A claim matching none of these carries no actionable security
# information regardless of how it is phrased.
RE_CVE = re.compile(r"\bCVE-\d{4}-\d{4,}\b", re.I)
RE_CWE = re.compile(r"\bCWE-\d{1,4}\b", re.I)
RE_ATTACK = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")
RE_TACTIC = re.compile(r"\bTA\d{4}\b")
RE_CONTROL = re.compile(r"\b(?:AC|AU|AT|CM|CP|IA|IR|MA|MP|PE|PL|PS|RA|SA|SC|SI|SR)-\d{1,2}\b")
RE_PORT = re.compile(r"\b(?:port|tcp|udp)[\s/:]*\d{1,5}\b", re.I)
RE_PATH = re.compile(r"(?:^|\s)(?:/[\w.\-]+){2,}|[A-Za-z]:\\\\")
RE_CMD = re.compile(r"(?:^|[\s`'\"])(?:sudo\s+)?[a-z0-9_.-]+\s+-{1,2}[a-zA-Z]")
RE_HTTPHDR = re.compile(r"\b(?:X-[\w-]+|Content-Security-Policy|Strict-Transport-Security|"
                        r"Set-Cookie|Authorization|Access-Control-Allow-\w+)\b", re.I)
RE_PROTO = re.compile(r"\b(?:HTTP/[12](?:\.\d)?|TLS\s?1\.[0-3]|SSLv[23]|SMBv[123]|"
                      r"IMDSv[12]|OAuth\s?2|SAML\s?2)\b", re.I)

ENTITY_PATTERNS = (RE_CVE, RE_CWE, RE_ATTACK, RE_TACTIC, RE_CONTROL, RE_PORT,
                   RE_PATH, RE_CMD, RE_HTTPHDR, RE_PROTO)

# Word-level security signal for text-based purity scoring.
_TEXT_TERMS = sorted({t.replace("-", "[- ]?") for t in SECURITY_TAGS if len(t) > 3},
                     key=len, reverse=True)
RE_SEC_TEXT = re.compile(r"\b(?:" + "|".join(_TEXT_TERMS) + r")\b", re.I)


def has_security_tag(rec: dict) -> bool:
    return bool({str(t).strip().lower() for t in (rec.get("tags") or [])} & SECURITY_TAGS)


def has_security_text(rec: dict) -> bool:
    blob = f"{rec.get('subject', '')} {rec.get('heading', '')} {rec.get('claim', '')}"
    return bool(RE_SEC_TEXT.search(split_identifiers(blob)))


def split_identifiers(text: str) -> str:
    """Break snake_case and camelCase so terms inside identifiers are visible to
    word-boundary matching. `\b` does not break on `_`, so DEFAULT_THROTTLE_CLASSES
    otherwise hides "throttle" and the claim reads as non-security."""
    text = text.replace("_", " ")
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)


def is_security(rec: dict) -> bool:
    return has_security_tag(rec) or has_security_text(rec)


def has_entity(claim: str) -> bool:
    """True if the claim names a concrete security entity: a CVE/CWE/technique,
    a control ID, a port, a path, a command with flags, an HTTP header, a protocol."""
    return any(p.search(claim) for p in ENTITY_PATTERNS)


# Identity-bearing entities. Two claims whose entity sets differ are distinct
# FACTS even when their surface form is near-identical, so near-dup collapse
# must never merge them. Without this guard SimHash deletes real data:
#   "npm package pnpm is affected by CVE-2026-42034: axios vulnerability."
#   "npm package cypress is affected by CVE-2024-43788: webpack vulnerability."
# share a template but name different packages and different CVEs.
RE_VERSION = re.compile(r"\b\d+\.\d+(?:\.\d+)*\b")
RE_QUOTED = re.compile(r"[`'\"]([A-Za-z0-9_.@/\-]{2,40})[`'\"]")

_ID_PATTERNS = (RE_CVE, RE_CWE, RE_ATTACK, RE_TACTIC, RE_CONTROL, RE_VERSION)


def entity_key(subject: str, claim: str) -> frozenset[str]:
    """Identity fingerprint of a claim: its subject plus every concrete
    identifier it names. Used as a hard guard on near-duplicate collapse."""
    ents = {m.group(0).upper() for pat in _ID_PATTERNS for m in pat.finditer(claim)}
    ents |= {m.group(1).lower() for m in RE_QUOTED.finditer(claim)}
    subj = re.sub(r"\W+", " ", (subject or "").lower()).strip()
    if subj:
        ents.add("subj:" + subj)
    return frozenset(ents)
