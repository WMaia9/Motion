"""Full unchanged font/margin/reference checks for the direction-control revision."""
from pathlib import Path
import audit_mardm_motion_layout as audit


def main():
    audit.R=Path('reports/cvpr_coherence_review_20261014')
    audit.main()


if __name__=='__main__':main()
