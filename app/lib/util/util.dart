class Grip {
  final int classId;
  final double centerX;
  final double centerY;
  final Map detection;

  Grip({
    required this.classId,
    required this.centerX,
    required this.centerY,
    required this.detection,
  });

  factory Grip.fromJson(Map<String, dynamic> json) {
    final List<dynamic> bbox = json['bbox'];
    double centerX = (((bbox[0] as num) + (bbox[2] as num)) / 2).toDouble();
    double centerY = (((bbox[1] as num) + (bbox[3] as num)) / 2).toDouble();
    return Grip(
      classId: json['class'] as int,
      centerX: centerX,
      centerY: centerY,
      detection: json,
    );
  }
}