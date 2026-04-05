import React from 'react'
import { View, Text, StyleSheet } from 'react-native'
import { DISEASE_INFO } from '../constants/models'

function ProbBar({ label, value, isTop }) {
  return (
    <View style={styles.probRow}>
      <Text style={styles.probLabel} numberOfLines={1}>{label}</Text>
      <View style={styles.track}>
        <View style={[styles.bar, isTop && styles.barTop, { width: `${value}%` }]} />
      </View>
      <Text style={styles.probPct}>{value.toFixed(1)}%</Text>
    </View>
  )
}

export default function ResultCard({ result }) {
  if (!result) return null
  const { predicted_class, confidence, all_probabilities } = result
  const info = DISEASE_INFO[predicted_class] ?? {
    displayName: predicted_class, emoji: '🌿', severity: '—', color: '#64748b',
    description: '', treatment: '',
  }

  const sorted = Object.entries(all_probabilities).sort((a, b) => b[1] - a[1])

  return (
    <View style={styles.card}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.emoji}>{info.emoji}</Text>
        <View style={styles.headerText}>
          <Text style={styles.className}>{info.displayName}</Text>
          <View style={styles.badges}>
            <Text style={styles.confBadge}>{confidence.toFixed(1)}% confidence</Text>
            <Text style={[styles.severity, { color: info.color }]}>{info.severity}</Text>
          </View>
        </View>
      </View>

      {/* Description */}
      {info.description ? (
        <View style={styles.section}>
          <Text style={styles.sectionText}>{info.description}</Text>
        </View>
      ) : null}

      {/* Treatment */}
      {info.treatment ? (
        <View style={[styles.section, styles.treatmentSection]}>
          <Text style={styles.sectionLabel}>RECOMMENDED TREATMENT</Text>
          <Text style={styles.sectionText}>{info.treatment}</Text>
        </View>
      ) : null}

      {/* Probabilities */}
      <View style={styles.section}>
        <Text style={styles.sectionLabel}>ALL CLASS PROBABILITIES</Text>
        <View style={styles.probList}>
          {sorted.map(([cls, prob]) => (
            <ProbBar
              key={cls}
              label={DISEASE_INFO[cls]?.displayName ?? cls}
              value={prob}
              isTop={cls === predicted_class}
            />
          ))}
        </View>
      </View>
    </View>
  )
}

const styles = StyleSheet.create({
  card: {
    borderRadius: 14,
    borderWidth: 1.5,
    borderColor: '#e2e8f0',
    backgroundColor: '#fff',
    overflow: 'hidden',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 14,
    padding: 16,
    backgroundColor: '#f0fdf4',
    borderBottomWidth: 1,
    borderBottomColor: '#b7e4c7',
  },
  emoji: { fontSize: 38 },
  headerText: { flex: 1 },
  className: { fontSize: 18, fontWeight: '700', color: '#1b4332', marginBottom: 4 },
  badges: { flexDirection: 'row', alignItems: 'center', gap: 10, flexWrap: 'wrap' },
  confBadge: {
    fontSize: 12, fontWeight: '700', color: '#40916c',
    backgroundColor: '#dcfce7', paddingHorizontal: 8, paddingVertical: 2, borderRadius: 999,
  },
  severity: { fontSize: 11, fontWeight: '700', textTransform: 'uppercase', letterSpacing: 0.5 },
  section: {
    padding: 14,
    borderBottomWidth: 1,
    borderBottomColor: '#f1f5f9',
  },
  treatmentSection: { backgroundColor: '#fffbeb' },
  sectionLabel: {
    fontSize: 10, fontWeight: '700', letterSpacing: 0.8, color: '#94a3b8',
    marginBottom: 6, textTransform: 'uppercase',
  },
  sectionText: { fontSize: 13, lineHeight: 19, color: '#1e293b' },
  probList: { gap: 10, marginTop: 4 },
  probRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  probLabel: { width: 110, fontSize: 12, fontWeight: '500', color: '#1e293b' },
  track: {
    flex: 1, height: 9, backgroundColor: '#e2e8f0', borderRadius: 999, overflow: 'hidden',
  },
  bar: { height: '100%', borderRadius: 999, backgroundColor: '#74c69d' },
  barTop: { backgroundColor: '#40916c' },
  probPct: { width: 40, fontSize: 11.5, fontWeight: '600', color: '#64748b', textAlign: 'right' },
})
