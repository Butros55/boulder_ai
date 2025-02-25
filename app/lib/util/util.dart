import 'dart:math';

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
}

List<List<Grip>> groupGripsIntoRoutes(
  List<dynamic> detections,
  double threshold, {
  Set<int> ignoredClassIds = const {},
  int minGripCount = 3, // Default: min. 3 Griffe pro Route
}) {
  // 1) Alle Griffe einsammeln, außer die ignorierten Klassen
  List<Grip> allGrips = [];
  for (var d in detections) {
    final bbox = d["bbox"] as List<dynamic>;
    double x1 = bbox[0].toDouble();
    double y1 = bbox[1].toDouble();
    double x2 = bbox[2].toDouble();
    double y2 = bbox[3].toDouble();

    double centerX = (x1 + x2) / 2;
    double centerY = (y1 + y2) / 2;
    int classId = d["class"] ?? 0;

    // Überspringen, wenn Klasse ignoriert
    if (ignoredClassIds.contains(classId)) {
      continue;
    }

    allGrips.add(Grip(
      classId: classId,
      centerX: centerX,
      centerY: centerY,
      detection: d,
    ));
  }

  // 2) Nach Farbe gruppieren
  Map<int, List<Grip>> gripsByColor = {};
  for (var grip in allGrips) {
    gripsByColor.putIfAbsent(grip.classId, () => []).add(grip);
  }

  // 3) Für jede Farbgruppe: zusammenhängende Komponenten suchen
  List<List<Grip>> routes = [];
  for (var group in gripsByColor.values) {
    routes.addAll(_findConnectedComponents(group, threshold));
  }

  // 4) Routen filtern: minGripCount
  List<List<Grip>> filteredRoutes = [];
  for (var route in routes) {
    if (route.length >= minGripCount) {
      filteredRoutes.add(route);
    }
  }

  return filteredRoutes;
}

List<List<Grip>> _findConnectedComponents(List<Grip> grips, double threshold) {
  List<List<Grip>> components = [];
  Set<int> visited = {};
  for (int i = 0; i < grips.length; i++) {
    if (!visited.contains(i)) {
      List<Grip> component = [];
      _dfs(i, grips, threshold, visited, component);
      components.add(component);
    }
  }
  return components;
}

void _dfs(int index, List<Grip> grips, double threshold, Set<int> visited, List<Grip> component) {
  visited.add(index);
  component.add(grips[index]);

  for (int j = 0; j < grips.length; j++) {
    if (!visited.contains(j)) {
      double dx = grips[index].centerX - grips[j].centerX;
      double dy = grips[index].centerY - grips[j].centerY;
      double distance = sqrt(dx * dx + dy * dy);
      if (distance < threshold) {
        _dfs(j, grips, threshold, visited, component);
      }
    }
  }
}
