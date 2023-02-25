import math
import os
import sys
from PySide6.QtCore import QDir, Slot, QTimer, \
    Signal, QItemSelection, QMargins, QRectF, QLineF, QPointF, \
    QParallelAnimationGroup, QPropertyAnimation, QEasingCurve, \
    QSize, Qt, QItemSelectionModel
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QPolygonF, QFont, \
    QWheelEvent, QPainterPath, QVector2D
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, \
    QFileSystemModel, QLabel, QAbstractItemView, \
    QSlider, QPushButton, QStyle, QHBoxLayout, QSplitter, QVBoxLayout, \
    QGraphicsObject, QGraphicsItem, QStyleOptionGraphicsItem, QGraphicsScene, \
    QGraphicsView, QListView
from problem import load_problem, Problem, State, load_plan, Solution


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._problem = None
        self._selector = SelectorWidget()
        self._mapview = MapView()
        self._player = PlayerWidget()
        self._info = InfoWidget()
        self._playback = (0, False)
        self._solution = None

        margins = QMargins(0,0,0,0)        
        left = QWidget()
        layout = QVBoxLayout(left)
        layout.setSpacing(2)
        layout.setContentsMargins(margins)
        layout.addWidget(self._selector)
        layout.addWidget(self._info)
        
        right = QWidget()
        layout = QVBoxLayout(right)
        layout.setSpacing(2)
        layout.setContentsMargins(margins)
        layout.addWidget(self._mapview)
        layout.addWidget(self._player)
        
        splitter = QSplitter()
        splitter.setContentsMargins(margins)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 4)
        self.setCentralWidget(splitter)
        
        self._selector.path.connect(self.onPathSelected)
        self._player.playback.connect(self.onPlayback)
    
    @Slot(str)
    def onPathSelected(self, path: str):
        directory = path if os.path.isdir(path) else os.path.dirname(path)
        problem = load_problem(directory)
        if problem:
            self._setProblem(directory, problem)
        else:
            self._info.setSolution(None) 
    
    def _setProblem(self, directory: str, problem: Problem):
        self._player.onStop()
        plan = load_plan(directory)
        self._solution = Solution(problem, plan)
        self._mapview.setProblem(problem)
        self._player.setMaxTime(plan.length())
        self._info.setSolution(self._solution)
    
    @Slot(int, bool)
    def onPlayback(self, time: int, playing: bool):
        old_time, old_play = self._playback
        self._playback = (time, playing)
        state = self._solution.state(time)
        if playing and old_time + 1 == time:
            self._mapview.animateState(state)
        elif not (old_play and not playing and old_time == time):
            self._mapview.setState(state)

 
class SelectorWidget(QListView):
    path = Signal(str)
    def __init__(self):
        super().__init__()
        model = QFileSystemModel()
        path = os.path.join(os.path.dirname(__file__), 'problems')
        model.setRootPath(path)
        model.setFilter(QDir.AllDirs | QDir.NoDotAndDotDot)
        self.setModel(model)
        self.setRootIndex(self.model().index(path))
        self.setMinimumSize(100, 100)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.selectionModel().selectionChanged.connect(self.onSelectionChanged)
    
    def sizeHint(self) -> QSize:
        return QSize(200,200)
    
    @Slot(QItemSelection, QItemSelection)
    def onSelectionChanged(self, selected: QItemSelection, deselected: QItemSelection):
        data = selected.data()
        if data:
            index = data.indexes()[0]
            path = self.model().filePath(index)
            self.path.emit(path)
        
    def select(self, path: str):
        path = os.path.join(self.model().rootPath(), path)
        index = self.model().index(path)
        self.selectionModel().select(index, QItemSelectionModel.SelectCurrent)

class InfoWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(100, 100)
        self._name = QLabel('')
        self._summary = QLabel('')
        self._score = QLabel('')
        self._validation = QLabel('')
        self._validation.setWordWrap(True)
        self._validation.setStyleSheet('QLabel { color : #ff0000; }')
        self._name.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self._validation.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.setSolution(None)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(2)
        layout.setContentsMargins(QMargins(2,2,2,2))
        layout.addWidget(self._name)
        layout.addWidget(self._summary)
        layout.addWidget(self._score)
        layout.addWidget(self._validation)
        layout.addStretch(1)

    def sizeHint(self) -> QSize:
        return QSize(220,200)
    
    def setSolution(self, solution: Solution):
        self._solution = solution
        if self._solution:
            self._name.setText(f'Problem: {self._solution.problem.name}')
            self._score.setText(f'Score: {self._solution.score():.2f}')
            self._update_validation()
        else:
            self._name.setText('No problem selected.')
            self._score.setText('')
            self._validation.setText('Choose a directory, it will be checked if it contains a problem.')

    def _update_validation(self):
        self._validation.setText(self._solution.generateReport())


class PlayerWidget(QWidget):
    playback = Signal(int, bool)
    def __init__(self):
        super().__init__()
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setOrientation(Qt.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.setSingleStep(1)
        self._slider.setPageStep(1)
        self._slider.setTickInterval(1)
        self._slider.setTickPosition(QSlider.TicksBothSides)
        self._play_pause = QPushButton(self.style().standardIcon(QStyle.SP_MediaPlay), "")
        self._stop = QPushButton(self.style().standardIcon(QStyle.SP_MediaStop), "")
        self._timeLabel = QLabel("0")
        self._timer = QTimer()
        self._time = 0
        self._playing = False
        self._isSliderPressed = False
        
        layout = QHBoxLayout(self)
        layout.addWidget(self._play_pause)
        layout.addWidget(self._stop)
        layout.addWidget(self._slider)
        layout.addWidget(self._timeLabel)
        
        self._slider.valueChanged.connect(self.onSliderValueChanged)
        self._slider.sliderPressed.connect(self.onSliderPressed)
        self._slider.sliderReleased.connect(self.onSliderReleased)
        self._play_pause.clicked.connect(self.onPlayPause)
        self._stop.clicked.connect(self.onStop)
        self._timer.timeout.connect(self.onTimerTick)
        self.playback.connect(self.onPlayback)
    
    def setMaxTime(self, time: int):
        self._slider.setMaximum(time)
        
    @Slot()
    def onTimerTick(self):
        time = self._time + 1
        if time > self._slider.maximum():
            self.onPlayPause()
        else:
            self._setPlayback(time, True)
    
    @Slot()
    def onPlayPause(self):
        if self._playing:
            self._timer.stop()
            self._setPlayback(self._time, False)
        elif self._time < self._slider.maximum():
            self._setPlayback(self._time + 1, True)
            self._timer.start(1000)
    
    @Slot()
    def onStop(self):
        self._timer.stop()
        self._setPlayback(0, False)
    
    @Slot(int)
    def onSliderValueChanged(self, t):
        self._setPlayback(t, self._playing)

    @Slot()
    def onSliderPressed(self):
        self._isSliderPressed = True
        self._wasPlaying = self._playing
        self._timer.stop()
        self._setPlayback(self._time, False)

    @Slot()
    def onSliderReleased(self):
        self._isSliderPressed = False
        if self._wasPlaying:
            self._setPlayback(self._time, True)
            self._timer.start(1000)
    
    @Slot(int, bool)
    def onPlayback(self, time: int, playing: bool):
        if not self._isSliderPressed:
            self._slider.setValue(time)
        icon = QStyle.SP_MediaPause if playing else QStyle.SP_MediaPlay
        self._play_pause.setIcon(self.style().standardIcon(icon))
    
    def _setPlayback(self, t: int, p: bool):
        if self._time != t or self._playing != p:
            self._time = t
            self._playing = p
            self.playback.emit(t, p)
            self._timeLabel.setText(str(t));
        

SIZE = 64
FONT = QFont('Arial', 10)

class MapView(QGraphicsView):
    
    def __init__(self):
        super().__init__()
        self._items = dict()
        self._animations = QParallelAnimationGroup()
        self._problem = None
        self.setScene(QGraphicsScene(self))
        self.scene().setBackgroundBrush(QBrush(QColor('white')))
        self.setMinimumSize(250, 250)
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.setResizeAnchor(QGraphicsView.NoAnchor)
    
    def sizeHint(self)-> QSize:
        return QSize(800, 600)
    
    def wheelEvent(self, event: QWheelEvent )->None:
        oldpos = self.mapToScene(event.position().toPoint())
        zoom = 1.15 if event.angleDelta().y() > 0 else 1/1.15
        self.scale(zoom, zoom)
        delta = self.mapToScene(event.position().toPoint()) - oldpos
        self.translate(delta.x(), delta.y())


    def setProblem(self, problem: Problem):
        self.scene().clear()
        self._items.clear()
        self._problem = problem
        
        for node in problem.graph.nodes(data=True):
            item = NodeItem(node)
            self._items[item.name] = item
            self.scene().addItem(item)
        for edge in problem.graph.edges:
            source = self._items[edge[0]]
            target = self._items[edge[1]]
            self.scene().addItem(EdgeItem(source, target))
        
        for agent in problem.agents:
            item = AgentItem(agent)
            item.setPos(self._calcAgentPos(problem.state, agent))
            self._items[item.name] = item
            self.scene().addItem(item)
        
        for box in problem.boxes:
            item = BoxItem(box)
            item.setPos(self._calcBoxPos(problem.state, box))
            self._items[item.name] = item
            self.scene().addItem(item)
            
        # The scene grows, but does not shrink automatically. Resize before centering. 
        self.setSceneRect(self.scene().itemsBoundingRect())
        # Reset the transformation matrix, which centers and scales back to 1.0
        self.resetTransform()
    
    def setState(self, state: State):
        self._animations.stop()
        self._animations = QParallelAnimationGroup()
        for agent in self._problem.agents:
            item = self._items[agent]
            item.setPos(self._calcAgentPos(state, agent))
            
        for box in self._problem.boxes:
            item = self._items[box]
            item.setPos(self._calcBoxPos(state, box))
    
    def animateState(self, state: State):
        self._animations.stop()
        self._animations = QParallelAnimationGroup()
        for agent in self._problem.agents:
            self._animatePosition(self._items[agent], self._calcAgentPos(state, agent))
        for box in self._problem.boxes:
            self._animatePosition(self._items[box], self._calcBoxPos(state, box))
        self._animations.start()
    
    def _animatePosition(self, item, pos):
        if item.pos() != pos:
            animation = QPropertyAnimation(item, b"pos")
            animation.setDuration(900)
            animation.setEasingCurve(QEasingCurve.InOutCubic)
            animation.setEndValue(pos)
            self._animations.addAnimation(animation)
    
    def _calcAgentPos(self, state: State, agent: str):
        return self._items[state[agent]].pos() + QPointF(0, SIZE) * 0.08
    
    def _calcBoxPos(self, state: State, box: str):
        location = state[box]
        if location is None:
            index = [b for b in self._problem.boxes if state[b] is None].index(box)
            return QPointF(-1.5 * SIZE, index * SIZE * 0.4)
        if location in self._problem.agents:
            return self._calcAgentPos(state, location) + QPointF(SIZE, 0) * 0.1
        else:
            return self._items[location].pos() + QPointF(SIZE, -SIZE) * 0.16


class NodeItem(QGraphicsObject):
    def __init__(self, node):
        super().__init__()
        self.name = node[0]
        self._size = SIZE * 0.75
        self._thickness = 1
        self._rect = QRectF(-self._size/2, -self._size/2, self._size, self._size)
        self._color = QColor("black")
        self.setPos(node[1]['col'] * SIZE, node[1]['row'] * SIZE)
        self.setCacheMode(QGraphicsItem.DeviceCoordinateCache)

    def boundingRect(self) -> QRectF:
        return self._rect.adjusted(-self._thickness, -self._thickness, self._thickness, self._thickness)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget = None):
        painter.setRenderHints(QPainter.Antialiasing)
        painter.setPen(QPen(self._color, self._thickness, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawRect(self._rect)
        painter.setFont(FONT)
        painter.drawText(self._rect.adjusted(1, 0, 0, 0), Qt.AlignLeft | Qt.AlignTop, self.name)

    def bordercut(self, point: QPointF) -> QPointF:
        '''Calculates the intersection point of this nodes border with the line through (self.center, point)'''
        line: QPointF = point - self.pos()
        maxcoordinate = max(math.fabs(line.x()), math.fabs(line.y()))
        return self.pos() + self._size / (2*maxcoordinate) * line


class EdgeItem(QGraphicsObject):
    def __init__(self, source: NodeItem, target: NodeItem):
        super().__init__()
        self._arrowsize =  0.05 * SIZE
        self._color = QColor("black")
        self._thickness = 2
        
        vector = QVector2D(target.pos() - source.pos()).normalized().toPointF()
        arrowstart = source.bordercut(target.pos()) + vector
        arrowend = target.bordercut(source.pos()) - vector
        arrowbase = arrowend - self._arrowsize * vector
        arrownormal = vector.transposed() * self._arrowsize * 2/3
        self.line = QLineF(arrowstart, arrowend)
        self.head = QPolygonF()
        self.head.append(arrowend)
        self.head.append(arrowbase + arrownormal)
        self.head.append(arrowbase - arrownormal)
        self.setZValue(-1)

    def boundingRect(self) -> QRectF:
        margin = self._thickness + self._arrowsize
        return QRectF(self.line.p1(), self.line.p2()).normalized().adjusted(-margin,-margin, margin, margin)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget=None):
        painter.setRenderHints(QPainter.Antialiasing)
        painter.setPen(QPen(self._color, self._thickness, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawLine(self.line)
        painter.setBrush(QBrush(self._color))
        painter.drawPolygon(self.head)


class AgentItem(QGraphicsObject):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self._hsize = SIZE * 0.65
        self._vsize = SIZE * 0.45
        self._radius = SIZE * 0.125
        self._thickness = 2
        self._rect = QRectF(-self._hsize/2, -self._vsize/2, self._hsize, self._vsize)
        self._color = QColor("#f25c19")
        self.setCacheMode(QGraphicsItem.DeviceCoordinateCache)
        self.setZValue(1)

    def boundingRect(self) -> QRectF:
        return self._rect.adjusted(-self._thickness, -self._thickness, self._thickness, self._thickness)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget = None):
        painter.setRenderHints(QPainter.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(self._rect, self._radius, self._radius)
        painter.fillPath(path, QBrush(self._color))
        painter.setPen(QPen(self._color.darker(125), self._thickness, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawPath(path)
        painter.setPen(QPen(QColor("white")))
        painter.setFont(FONT)
        painter.drawText(self._rect.adjusted(1, 0, 0, 0), Qt.AlignLeft | Qt.AlignTop, self.name)


class BoxItem(QGraphicsObject):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self._size = SIZE * 0.33
        self._rect = QRectF(-self._size/2, -self._size/2, self._size, self._size)
        self._thickness = 2
        self._color = QColor("#195cf2")
        self.setCacheMode(QGraphicsItem.DeviceCoordinateCache)
        self.setZValue(2)

    def boundingRect(self) -> QRectF:
        return self._rect.adjusted(-self._thickness, -self._thickness, self._thickness, self._thickness)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget = None):
        painter.setRenderHints(QPainter.Antialiasing)
        painter.setPen(QPen(self._color.darker(125), 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(QBrush(self._color))
        painter.drawRect(self._rect)
        painter.setPen(QPen(QColor("white")))
        painter.setFont(FONT)
        painter.drawText(self._rect, Qt.AlignCenter , self.name)


if __name__ == '__main__':
    print('sys.argv: ', sys.argv)
    app = QApplication(sys.argv)
    window = MainWindow()
    window._selector.select("swap")
    window.resize(1280, 720)
    window.show()
    
    status = app.exec()
    sys.exit(status)
