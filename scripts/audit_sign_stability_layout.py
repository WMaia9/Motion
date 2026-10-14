"""Full native font/margin/reference checks for the sign-stability delivery."""
from pathlib import Path
import audit_mardm_motion_layout as audit


def main():
    audit.R=Path('reports/cvpr_sign_stability_review_20261014')
    audit.main()


if __name__=='__main__':main()
