from flask import Blueprint, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from models.models import Analysis

analysis_bp = Blueprint('analysis_bp', __name__)

@analysis_bp.route("/analyses", methods=["GET"])
@jwt_required()
def get_analyses():
    user_id = get_jwt_identity()
    analyses = (Analysis.query.filter_by(user_id=user_id)
                .order_by(Analysis.timestamp.desc())
                .limit(5)
                .all())
    total_count = Analysis.query.filter_by(user_id=user_id).count()

    data = []
    for analysis in analyses:
        data.append({
            "id": analysis.id,
            "original_image": analysis.original_image,
            "detection_image": analysis.detection_image,
            "timestamp": analysis.timestamp.isoformat()
        })
    return jsonify({
        "total": total_count,
        "analyses": data
    })
